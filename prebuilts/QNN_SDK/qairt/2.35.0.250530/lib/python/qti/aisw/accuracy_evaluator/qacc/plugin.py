# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import builtins
from collections import OrderedDict
import copy
import glob
import importlib
import inspect
import os
import shutil
import sys
from joblib import Parallel, delayed
from pathlib import Path
import re
import qti.aisw.accuracy_evaluator.common.exceptions as ce
import qti.aisw.accuracy_evaluator.qacc.reader as rd
import qti.aisw.accuracy_evaluator.qacc.writer as wr
from qti.aisw.accuracy_evaluator.qacc import *
from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.qacc import qacc_file_logger, qacc_logger
# Max number of input images.
# TODO - replace with BATCH_SIZE when batch mode pipeline is implemented.

batched_inputs_idx_parser = re.compile("-(\d+)_(\d+)-")
single_inputs_idx_parser = re.compile("-(\d+)-\d+")
#TODO: Update with Current Ref platform infer out
ref_single_outputs_idx_parser = re.compile("\w+_(\d+)\.")


class PipelineCache:
    """Class acting as global pipeline_cache for the entire pipeline and
    plugins to share relevant information between the plugins or stages of the
    E2E pipeline."""
    __instance = None

    def __init__(self):
        if PipelineCache.__instance is not None:
            pass
        else:
            PipelineCache.__instance = self
            self._pipeline_cache = {}
            self._nested_keys = [
                qcc.PIPELINE_POSTPROC_DIR, qcc.PIPELINE_POSTPROC_FILE, qcc.PIPELINE_INFER_DIR,
                qcc.PIPELINE_INFER_FILE, qcc.PIPELINE_NETWORK_DESC, qcc.PIPELINE_NETWORK_BIN_DIR,
                qcc.PIPELINE_PROGRAM_QPC, qcc.INTERNAL_INFER_TIME, qcc.INTERNAL_POSTPROC_TIME,
                qcc.INTERNAL_METRIC_TIME, qcc.INTERNAL_QUANTIZATION_TIME,
                qcc.INTERNAL_COMPILATION_TIME
            ]
            # init empty nested keys
            for key in self._nested_keys:
                self._pipeline_cache[key] = {}

    @classmethod
    def getInstance(cls):
        if PipelineCache.__instance is None:
            PipelineCache()
        return cls.__instance

    def set_val(self, key, val, nested_key=None):
        """Stores the key and value in the global dictionary."""
        if nested_key is None:
            self._pipeline_cache[key] = val
            qacc_file_logger.debug('Pipeline pipeline_cache - storing key {} value {}'.format(
                key, val))
        else:
            self._pipeline_cache[key][nested_key] = val
            qacc_file_logger.debug('Pipeline pipeline_cache - storing key {}:{}  value {}'.format(
                key, nested_key, val))

    def get_val(self, key, nested_key=None):
        """Returns value from information stored in dictionary during various
        stages of the pipeline.

        Args:
            key_string: nested keys in string format eg key.key.key

        Returns:
            value: value associated to the key, None otherwise
        """
        if key not in self._nested_keys:
            if key in self._pipeline_cache:
                return self._pipeline_cache[key]
            else:
                qacc_file_logger.warning('Pipeline pipeline_cache key {} not found'.format(key))
        else:
            if key in self._pipeline_cache and nested_key in self._pipeline_cache[key]:
                return self._pipeline_cache[key][nested_key]
            else:
                qacc_file_logger.warning('Pipeline pipeline_cache key {}:{} not found'.format(
                    key, nested_key))
        return None


def pl_print(*args, **kwargs):
    # TODO - change to debug
    qacc_file_logger.info(*args)


def _parse_range(index_str):
    if len(index_str) == 0:
        return []
    nums = index_str.split("-")
    assert len(nums) <= 2, 'Invalid range in calibration file '
    start = int(nums[0])
    end = int(nums[-1]) + 1
    return range(start, end)


# Dump all prints to qacc log file.
builtins.print = pl_print


class Plugin:
    """QACC Plugin class."""

    def __init__(self, plugin_config):
        self._plugin_config = plugin_config


class PluginManager:
    """QACC Plugin Manager class This class is responsible for running a set of
    plugins during preprocessing, postprocessing and metrics calculation."""
    registered_plugins = {}
    registered_metric_plugins = {}
    registered_dataset_plugins = {}

    def __init__(self, dataset=None, orig_dataset=None):
        # This dataset object is needed for inputlist file during preprocessing and
        # annotation file during post processing
        # dataset is None in case of dataset plugins as dataset object is
        # instantiated post dataset transformation execution.
        self.dataset = dataset
        self.orig_dataset = orig_dataset
        self.reader = rd.Reader()
        self.writer = wr.Writer()

    @classmethod
    def findAvailablePlugins(cls,):
        qacc_file_logger.info('Searching for plugins.')
        # find all plugin classes in the default plugin location and the given custom path, if it exists.
        plugin_path = str(Path(__file__).resolve().parents[1] / 'plugins/filebased_plugins')
        all_plugin_paths = [plugin_path]
        custom_plugin_path = os.getenv('CUSTOM_PLUGIN_PATH')
        if custom_plugin_path:
            if os.path.exists(custom_plugin_path):
                all_plugin_paths.append(custom_plugin_path)
            else:
                qacc_file_logger.warning(
                    f"Custom plugin path {custom_plugin_path} does not exist. No custom plugins would be loaded"
                )

        for path in all_plugin_paths:
            abs_path = os.path.abspath(path)
            sys.path.append(abs_path)
            # add to sys path
            dirs = []
            for dir in os.walk(abs_path):
                if dir[0] not in dirs:
                    sys.path.append(dir[0])
                    dirs.append(dir[0])

            # search plugins recursively
            paths = glob.glob(path + '/**/' + '*.py', recursive=True)
            files = [file.rsplit('/', 1)[-1] for file in paths]
            for idx, file in enumerate(files):
                if file.endswith(".py"):
                    if file.startswith('__'):
                        continue
                    file = os.path.splitext(file)[0]
                    _plugin = importlib.import_module(file)
                    classes = inspect.getmembers(_plugin, predicate=inspect.isclass)
                    for cl in classes:
                        if cl[1].__module__ == file:
                            class_hier = inspect.getmro(cl[1])
                            for class_h in class_hier:
                                if class_h.__name__ == 'qacc_plugin':
                                    PluginManager.registered_plugins[cl[0]] = cl[1]
                                    qacc_file_logger.info('Registered plugin :' + cl[0])
                                    break
                                elif class_h.__name__ == 'qacc_metric':
                                    PluginManager.registered_metric_plugins[cl[0]] = cl[1]
                                    qacc_file_logger.info('Registered metric plugin :' + cl[0])
                                    break
                                elif class_h.__name__ == 'qacc_dataset':
                                    PluginManager.registered_dataset_plugins[cl[0]] = cl[1]
                                    qacc_file_logger.info('Registered dataset plugin :' + cl[0])
                                    break


    def execute_dataset_transformations(self, dataset_config, out_dir):
        plugin_list = dataset_config.dataset_plugin_list

        # create dataset directory
        if plugin_list is not None:
            if len(plugin_list) > 0:
                if not os.path.exists(out_dir):
                    os.makedirs(out_dir)

            # execute plugins
            for pl_idx, plugin in enumerate(plugin_list):
                status, dataset_config = self.execute_dataset_plugin(plugin, dataset_config,
                                                                     out_dir)
                if status:
                    qacc_file_logger.error('Error while executing dataset transformation')

        return dataset_config

    def execute_metrics(self, metrics_plugin_config, output_dir, results_str_list, results_dict):

        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for pl_cfg in metrics_plugin_config:
            plugin = Plugin(pl_cfg)
            m_in = MetricInputInfo(plugin._plugin_config,
                                   result_file=self.dataset.get_input_list_file(),
                                   orig_inp_file=self.orig_dataset.get_orig_input_list_file(),
                                   gt_file=self.orig_dataset.get_dataset_annotation_file(),
                                   out_dir=output_dir)

            # (self, plugin_config, paths, result_file, gt_file, out_dir):
            m_out = MetricResult(out_dir=output_dir)
            self.execute_metric_plugin(plugin, m_in, m_out)
            if m_out.status != 0:
                qacc_file_logger.error('Plugin ({}) failed to execute!!'.format(
                    plugin._plugin_config.name))
                return 1

            results_str_list.append(m_out.result_str)
            results_dict.update(m_out.result)
        return 0

    def get_chained_transformations(self, transformations):
        """
        Returns a list of thread safe transformation list
        eg:
            plugin1 (mem) -> plugin2 (path) -> plugin3 (dir) -> plugin4 (mem)
            should return a transformation list
            chained_transformations = [transformation1, transformation2, transformation3]
            transformation1: [Plugin1, Plugin2]
            transformation2: [Plugin3]
            transformation3: [Plugin4]
        Returns:
            chained_transformations: list of transformations
        """
        chained_transformations = []

        # flag checked before creating a new transformations instance
        is_new_chain = True

        for idx, plugin in enumerate(transformations.plugin_list):
            # create new transformation chain if this is start of
            # the transformations chain
            if is_new_chain is True:
                transformations = Transformations(max_input_len=self.dataset.get_total_entries())
                is_new_chain = False

            # input_info is a tuple (type, dtype, format)
            # check if the type is dir
            if plugin._plugin_config._input_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_DIR:
                # check if any earlier transformation chain is not empty
                if not transformations.is_empty():
                    # save old transformation
                    chained_transformations.append(copy.deepcopy(transformations))

                # add new dir based transformation
                # is_dir fields is marked as true to distinguish this transformation
                # as a directory based transformation
                chained_transformations.append(
                    copy.deepcopy(
                        Transformations(plugin_config_list=[plugin._plugin_config], is_dir=True,
                                        max_input_len=self.dataset.get_total_entries())))

                # delete existing transformations as it is already saved
                del transformations

                # create a new transformation in next iteration
                is_new_chain = True
            else:
                transformations.add_plugin(plugin)

        # add last iterated transformations if not empty
        # checking in locals to confirm var transformations
        # was not deleted previously
        if ('transformations' in locals()) and \
                (not transformations.is_empty()):
            # save old transformation
            chained_transformations.append(copy.deepcopy(transformations))

        return chained_transformations

    def execute_transformations(self, transformations, output_dir, batch_offset=0,
                                squash_results=False, input_names=None):
        """Executes all the plugins in the entire transformation chain The
        execute method calls chain transformation method To split the entire
        transformation into smaller transformation chains which can run in
        parallel.

        Args:
            transformations: a set of plugins to be performed in a sequence
            output_dir: used as the target path to store intermediate outputs
        """
        ret_status = 0
        # create output path if not existing.
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        # get thread safe changed transformations
        chained_transformations = self.get_chained_transformations(transformations)

        last_transform_outputs = None

        # iterate the chained transformations and execute them
        for tr_idx, transformations in enumerate(chained_transformations):
            # Execute the transformations based on the is_dir flag of the transformations
            # is_dir flag true shows that the transformations contains single directory based plugin
            # is_dir flag false shows the transformations contains non directory based plugins
            # which can be executed in parallel

            if transformations.is_dir is True:

                # dir_input is in a form of nested paths
                # e.g. [[path1.1], [path2.1]] or
                # [[path1.1, path1.2, path1.3], [path2.1, path2.2, path2.3] … and so on]
                if tr_idx == 0:
                    # If this is the first transformation, the inputs come from dataset.
                    dir_input_paths = [input_path for input_path in self.dataset.input_generator()]
                else:
                    # if not the first transformation, use the output paths from last
                    # transformation.
                    dir_input_paths = last_transform_outputs

                ret_status = self.execute_dir_plugin(transformations, dir_input_paths, output_dir)

                # save the last transformation output paths
                last_transform_outputs = transformations.output_path_list

            else:
                # create multiple threads to execute the plugins in parallel
                # The number of threads is dependents on the available cores in the machine.
                # the n_jobs=-1 tries to create maximum number of threads based on available cores
                # TODO set batch_offset for batch mode

                if last_transform_outputs:
                    # Transformation followed by directory plugin in pipeline.
                    generator = last_transform_outputs
                else:
                    # First transformation in the pipeline stage. Use dataset.
                    generator = self.dataset.input_generator()

                Parallel(n_jobs=-1, verbose=0, prefer="threads") \
                    (delayed(self.execute_transformation_in_parallel)(transformations,
                                                                      input_path,
                                                                      output_dir,
                                                                      input_index + batch_offset)
                     # input_path is a path or a set of paths
                     # e.g. [path1] or [path1, path2, ... pathN]
                     for input_index, input_path in enumerate(generator))

                if transformations.run_status:
                    qacc_file_logger.error('Plugin Transformation failed to run.')
                    ret_status = 1
                    break

                # save the last transformation output paths
                last_transform_outputs = transformations.output_path_list

        # remove any None entries from list
        transformations.output_path_list = [i for i in transformations.output_path_list if i]
        # All transformations have executed. Now create the output list file.
        with open(os.path.join(output_dir, 'processed-outputs.txt'), 'w') as fl:
            for inps in transformations.output_path_list:
                if squash_results:
                    # Append the contents of each file in outputs list file.
                    # Only enabled after postprocessing for merging text file contents of outputs.
                    assert len(inps) == 1, 'Multiple input paths not supported for squashing. ' \
                                           'Disable squashing using squash_results=False in ' \
                                           'postprocessing'
                    with open(inps[0], mode='r') as f:
                        fl.write(f.read())

                else:
                    for i, inp in enumerate(inps):
                        if inp is None:
                            qacc_file_logger.error('Null input found at index {} while creating'
                                                   ' processed-outputs.txt.\n Record {}'.format(
                                                       i, inps))
                            raise RuntimeError('Some inputs were not processed!')
                        if i:
                            fl.write(',' + inp)
                        else:
                            fl.write(inp)
                    fl.write('\n')

        #Write the processed inputs in a different format.
        if input_names is not None:
            with open(os.path.join(output_dir, 'qnn-processed-outputs.txt'), 'w') as fl:
                for inps in transformations.output_path_list:
                    for i, inp in enumerate(inps):
                        if inp is None:
                            qacc_file_logger.error('Null input found at index {} while creating'
                                                   ' qnn-processed-outputs.txt.\n Record {}'.format(
                                                       i, inps))
                            raise RuntimeError('Some inputs were not processed!')
                        inp_abspath = os.path.abspath(inp)
                        if i:
                            fl.write(f" {input_names[i]}:={inp_abspath}")
                        else:
                            fl.write(f"{input_names[i]}:={inp_abspath}")
                    fl.write('\n')

        return ret_status

    def execute_transformation_in_parallel(self, transformations, input_path_list, output_dir,
                                           input_index):
        """Executes a chain of plugins.

        This method runs in a joblib thread.
        """
        # maintain one output info for number of inputs per record
        idx_plugin_inp_info = [
            PluginInputInfo(path=path, out_dir=output_dir) for path in input_path_list
        ]
        idx_plugin_out_info = [PluginOutputInfo() for _ in range(len(input_path_list))]

        # get original file names for automatically creating output file names by plugins.
        fnames = []
        for input_name in input_path_list:
            _, fname = os.path.split(input_name)
            fnames.append(os.path.splitext(fname)[0])

        for plugin in transformations.plugin_list:

            # execute function based plugin

            # find indexes of input record which this plugin will be passed.
            # default - all the inputs in the input record.
            if plugin._plugin_config._indexes:
                indexes = plugin._plugin_config._indexes
            else:
                indexes = list(range(len(idx_plugin_inp_info)))

            # evaluate inp/out info for mem/path plugin for each index.
            # Note : Only the required indexes are updated. Older objects are preserved.
            for index in indexes:
                # creating plugin input info to be passed to execute plugin
                idx_plugin_inp_info[index] = self.get_plugin_input_info(
                    plugin, idx_plugin_inp_info[index], idx_plugin_out_info[index], input_index)

                idx_plugin_out_info[index] = self.get_plugin_output_info(
                    plugin, idx_plugin_out_info[index], fnames[index], output_dir)
            self.execute_function_plugin(plugin, idx_plugin_inp_info, idx_plugin_out_info)

            deleted_slots = []
            for index in indexes:
                if idx_plugin_out_info[index].status == qcc.STATUS_ERROR:
                    qacc_file_logger.error('Plugin ({}) failed to execute!!'.format(
                        plugin._plugin_config.name))
                    transformations.run_status = 1
                    return
                elif idx_plugin_out_info[index].status == qcc.STATUS_REMOVE:
                    deleted_slots.append(index)

            # If current plugin has disabled some slots, remove them from the chain.
            indexes_to_delete = sorted(deleted_slots, reverse=True)

            for index in indexes_to_delete:
                del idx_plugin_inp_info[index]
                del idx_plugin_out_info[index]
                del fnames[index]

        # Persists in memory data at end of transformation.
        out_paths = transformations.output_path_list[input_index]
        if out_paths is None:
            out_paths = [None] * len(idx_plugin_out_info)
            transformations.output_path_list[input_index] = out_paths

        for idx, pout in enumerate(idx_plugin_out_info):
            if pout.plugin_config:
                if pout.plugin_config._output_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_MEM:
                    self.writer.write(output_path=pout.get_output_path(), mem_obj=pout.mem_obj,
                                      dtype=pout.get_output_dtype(),
                                      write_format=pout.get_output_format())
                out_paths[idx] = pout.get_output_path()
            else:
                # This input index has not been processed by any of the plugins.
                # Copy the input as it is to the output dir.
                qacc_file_logger.info('Copied unprocessed input {} to output dir {}'.format(
                    input_path_list[idx], output_dir))
                shutil.copy(input_path_list[idx], output_dir)
                out_paths[idx] = input_path_list[idx]

    def get_plugin_output_info(self, plugin, cur_pout, fname, output_dir):

        # point to the new plugin
        cur_pout.plugin_config = plugin._plugin_config
        cur_pout.outdir = output_dir
        cur_pout.status = 1  # Plugins must set this to 0 to mark successful execution.
        cur_pout.fname = fname  # used to generate the same name output file.

        return cur_pout

    def get_plugin_input_info(self, plugin, cur_pinp, last_pout, input_index):

        input_info = plugin._plugin_config._input_info
        cur_pinp.plugin_config = plugin._plugin_config

        last_output_info = None
        if last_pout.plugin_config:
            # This indicates that it is not the first plugin in the chain.
            last_output_info = last_pout.plugin_config._output_info

        if last_output_info is None:
            # first plugin in chain
            if input_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_MEM:
                cur_pinp.mem_obj = self.reader.read(input_path=cur_pinp.path,
                                                    dtype=input_info[qcc.IO_DTYPE],
                                                    format=input_info[qcc.IO_FORMAT])
        else:
            # compare prev output format with current input format.
            if last_output_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_MEM \
                    and input_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_PATH:

                # check formats.
                assert last_output_info[qcc.IO_FORMAT] == input_info[qcc.IO_FORMAT], \
                    'Mem Plugin {} out format must match Path plugin {} input format for the' \
                    'same index'.format(last_pout.plugin_config.name, plugin._plugin_config.name)

                self.writer.write(output_path=last_pout.get_output_path(),
                                  mem_obj=last_pout.mem_obj, dtype=last_output_info[qcc.IO_DTYPE],
                                  write_format=last_output_info[qcc.IO_FORMAT])
                cur_pinp.path = last_pout.get_output_path()

            elif last_output_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_PATH and \
                    input_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_MEM:

                # check formats.
                assert last_output_info[qcc.IO_FORMAT] == input_info[qcc.IO_FORMAT], \
                    'Path Plugin {} out format must match Mem plugin {} input format for the' \
                    'same index'.format(last_pout.plugin_config.name, plugin._plugin_config.name)

                cur_pinp.mem_obj = self.reader.read(input_path=last_pout.get_output_path(),
                                                    dtype=input_info[qcc.IO_DTYPE],
                                                    format=input_info[qcc.IO_FORMAT])

            elif last_output_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_PATH and \
                    input_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_PATH:
                cur_pinp.path = last_pout.get_output_path()

            else:
                cur_pinp.mem_obj = last_pout.mem_obj

        # orig dataset input.
        if self.orig_dataset:
            cur_pinp.orig_dataset = self.orig_dataset
            cur_pinp.input_index = input_index

        return cur_pinp

    def execute_dataset_plugin(self, plugin, dataset_config, out_dir):

        def set_dataset_info(dataset_config, dataset_pl_out_info):

            def set_updated(original, updated):
                return updated if updated else original

            dataset_config.inputlist_file = set_updated(dataset_config.inputlist_file,
                                                        dataset_pl_out_info.inputlist_file)
            dataset_config.annotation_file = set_updated(dataset_config.annotation_file,
                                                         dataset_pl_out_info.annotation_file)
            dataset_config.calibration_file = set_updated(dataset_config.calibration_file,
                                                          dataset_pl_out_info.calibration_file)
            dataset_config.calibration_type = set_updated(dataset_config.calibration_type,
                                                          dataset_pl_out_info.calibration_type)

            # check if inputlist path is modified in dataset plugin, if yes, set base path to out_dir
            if dataset_pl_out_info.inputlist_path_modified:
                dataset_config._inputlist_path = dataset_pl_out_info.out_dir
            # check if calibration path is modified in dataset plugin, if yes, set base path to out_dir
            if dataset_pl_out_info.calibration_path_modified:
                dataset_config._calibration_path = dataset_pl_out_info.out_dir

            return dataset_config

        dataset_pl_in_info = DatasetPluginInputInfo(plugin=plugin, dataset_config=dataset_config)
        dataset_pl_out_info = DatasetPluginOutputInfo(out_dir=out_dir)

        # execute plugin
        plugin_name = plugin.name
        if plugin_name not in PluginManager.registered_dataset_plugins:
            raise ce.ConfigurationException('Invalid dataset plugin name {}.'.format(plugin_name))
        plugin_class = PluginManager.registered_dataset_plugins[plugin_name]
        try:
            plugin_class().execute(dataset_pl_in_info, dataset_pl_out_info)
            status = 0
        except Exception as e:
            qacc_logger.error(
                'Plugin: {} failed to execute. See qacc.log for more details.'.format(plugin_name))
            qacc_file_logger.exception('Exception: {}'.format(e))
            status = 1

        # read output info and update dataset config
        if not dataset_pl_out_info.status:
            dataset_config = set_dataset_info(dataset_config, dataset_pl_out_info)
            dataset_config.update_max_inputs()

            if plugin_name == qcc.DATASET_FILTER_PLUGIN_NAME:
                # set the max_inputs and max_calib in pipeline cache
                pipeline_cache = PipelineCache.getInstance()
                pipeline_cache.set_val(
                    qcc.PIPELINE_MAX_INPUTS,
                    dataset_pl_in_info.get_param(qcc.DATASET_FILTER_PLUGIN_PARAM_MAX_INPUTS, -1))
                pipeline_cache.set_val(
                    qcc.PIPELINE_MAX_CALIB,
                    dataset_pl_in_info.get_param(qcc.DATASET_FILTER_PLUGIN_PARAM_MAX_CALIB, -1))

        return status, dataset_config

    def execute_dir_plugin(self, transformations, dir_input_paths, output_dir):
        # Execute Dir plugin
        plugin = transformations.plugin_list[0]
        input_info = plugin._plugin_config._input_info
        assert input_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_DIR, \
            'Internal error: Expecting dir based plugin'

        # remove any None entries from list
        dir_input_paths = [i for i in dir_input_paths if i]

        # orig dataset
        orig_input_paths = None
        if self.orig_dataset:
            orig_input_paths = self.orig_dataset.get_all_records()

        plugin_input_info = PluginInputInfo(plugin_config=plugin._plugin_config, mem_obj=None,
                                            path=dir_input_paths, orig_dataset=self.orig_dataset,
                                            out_dir=output_dir)
        plugin_output_info = PluginOutputInfo(plugin_config=plugin._plugin_config, status=1,
                                              out_dir=output_dir)

        self.execute_function_plugin(plugin, [plugin_input_info], [plugin_output_info])
        if plugin_output_info.status != 0:
            qacc_file_logger.error('Plugin ({}) failed to execute!!'.format(
                plugin._plugin_config.name))
            return 1

        transformations.output_path_list = plugin_output_info.updated_path
        return 0

    def execute_function_plugin(self, plugin, plugin_input_info_list, plugin_output_info_list):
        plugin_name = plugin._plugin_config.name
        if plugin_name not in PluginManager.registered_plugins:
            raise ce.ConfigurationException('Invalid plugin name {}.'.format(plugin_name))
        plugin_class = PluginManager.registered_plugins[plugin_name]
        try:
            plugin_class().execute(plugin_input_info_list, plugin_output_info_list)
        except Exception as e:
            qacc_logger.error(
                'Plugin: {} failed to execute. See qacc.log for more details.'.format(plugin_name))
            qacc_file_logger.exception('Exception: {}'.format(e))
            plugin_output_info_list[0].status = qcc.STATUS_ERROR

    def execute_metric_plugin(self, plugin, metric_input_info, metric_result):
        plugin_name = plugin._plugin_config.name
        if plugin_name not in PluginManager.registered_metric_plugins:
            raise ce.ConfigurationException('Invalid metric plugin name {}.'.format(plugin_name))
        plugin_class = PluginManager.registered_metric_plugins[plugin_name]
        try:
            plugin_class().execute(metric_input_info, metric_result)
        except Exception as e:
            qacc_logger.error(
                'Plugin: {} failed to execute. See qacc.log for more details.'.format(plugin_name))
            qacc_file_logger.exception('Exception: {}'.format(e))
            metric_result.status = 1


class Transformations:
    """QACC transformation class."""

    def __init__(self, max_input_len, plugin_config_list=None, is_dir=False):
        """
        Args:
            plugin_config_list: list of plugin configs
            is_dir:
                - used to represent if the transformation has dir based plugin
                - it is used in get_chained_transformations method in PluginManager class
                to distinguish between a single directory and other plugin transformations
        """
        if plugin_config_list is None:
            self.plugin_list = []
        else:
            self.plugin_list = self.get_plugin_list_from_plugin_config_list(plugin_config_list)
        self.is_dir = is_dir
        self.run_status = 0
        # path of output files at the end of this transformation.
        # format [[p11,p12],[p21,p22]..], considering two inputs per input record
        self.output_path_list = [None] * max_input_len

    def add_plugin(self, plugin):
        """Adds plugins to the list of transformations."""
        self.plugin_list.append(plugin)

    def is_empty(self):
        return len(self.plugin_list) == 0

    def get_plugin_list_from_plugin_config_list(self, plugin_config_list):
        """Returns a list of plugin objects from the list of plugin config
        objects.

        Args:
            plugin_config_list: list of plugin configs

        Returns:
            plugin_list: list of plugin objects
        """
        plugin_list = []
        for plugin_config in plugin_config_list:
            plugin_list.append(Plugin(plugin_config))
            plugin_config.update_input_output_info()
        return plugin_list


class PluginInputInfo:
    """QACC Plugin Input Info class."""

    def __init__(self, plugin_config=None, mem_obj=None, path=None, orig_dataset=None,
                 out_dir=None):
        # Stores the entire plugin configuration
        self.plugin_config = plugin_config

        # In memory object for plugins that work on in memory objects
        # This is useful when multiple plugins are chained together and the output of one plugin
        # is fed as memory object to the next plugin
        self.mem_obj = mem_obj

        # For path based plugin. Also set in case of mem plugins (1D list)
        # For dir based plugins , its a list of list of input record lists. (2D list)
        self.path = path

        # points to original dataset having list of lists of paths for all the inputs.
        # Used mostly by postprocessors which needs original input records.
        self.orig_dataset = orig_dataset

        # input index of the data
        self.input_index = None

        # inference_schema key for pipeline pipeline_cache
        inference_schema = [s for s in out_dir.split('/') if s.startswith('schema')]
        self._inference_schema_name = inference_schema[0] if len(inference_schema) != 0 else None

        # used to store any user object to be transferred to plugins
        # in the chain.
        self.user_obj = None

    def set_user_obj(self, obj):
        self.user_obj = obj

    def get_user_obj(self):
        return self.user_obj

    def get_input_index(self):
        if not self.is_directory_input():
            return self.input_index
        else:
            qacc_file_logger.error('get_input_index() is not supported for directory plugin')
            return None

    def is_memory_input(self):
        return self.plugin_config._input_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_MEM

    def is_directory_input(self):
        return self.plugin_config._input_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_DIR

    def is_path_input(self):
        return self.plugin_config._input_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_PATH

    def get_input_dtype(self):
        return self.plugin_config._input_info[qcc.IO_DTYPE]

    def get_input(self):
        if self.is_memory_input():
            return self.mem_obj
        else:
            return self.path

    def get_orig_path_list(self, num_elements=1, last_batch=qcc.LAST_BATCH_NO_CHANGE):
        """Returns the original input record(s) (as a list of paths) linked to
        the current inference output(s) being processed by the post processor
        based on the num_elements. The last batch is handled using the
        last_batch parameter. Currently last_batch option supports 3 options if
        size of last batch is not equal to num_elements. The options are
        available in qacc constants.

        Example usage from plugins:
            import qacc.constants as qcc
            orig_path_list = pin.get_orig_path_list(num_elements, qcc.LAST_BATCH_TRUNCATE)

        Args:
            num_elements: number of elements to be returned per batch
            last_batch: options to handle the last batch when size of last batch is not equal to
            num_elements
                LAST_BATCH_TRUNCATE: Truncates last batch and returns empty nested list for the
                last batch
                LAST_BATCH_REPEAT_LAST_RECORD: Returns the last batch with last record repeated
                to match the size of num_elements
                LAST_BATCH_NO_CHANGE: Returns the last batch without repeating the last record
        """
        if self.orig_dataset:
            if self.is_memory_input() or self.is_path_input():
                orig_input_paths = self.orig_dataset.get_record(
                    idx=self.input_index, num_records=num_elements if num_elements else 1,
                    last_batch=last_batch)
            elif self.is_directory_input():
                orig_input_paths = self.orig_dataset.get_all_records(group=num_elements,
                                                                     last_batch=last_batch)
            else:
                qacc_file_logger.error('Invalid plugin input info type: {}'.format(
                    self.plugin_config._input_info[qcc.IO_TYPE]))
            return orig_input_paths
        else:
            return None

    def get_env_tag(self):
        return self.plugin_config._env

    def get_param(self, param, default=None):
        if self.plugin_config.params is None:
            return default
        if param in self.plugin_config.params:
            val = self.plugin_config.params[param]
            if isinstance(val, str):
                val = val.strip()
            return val
        return default

    def __str__(self):
        return 'Mem: {}. Plugin Info: {}'.format(self.is_memory_input(), self.plugin_config)

    def get_inference_schema_name(self):
        return self._inference_schema_name.split(
            '_')[1] if self._inference_schema_name is not None else None

    # API to access pipeline_cache
    def read_pipeline_cache_val(self, key):
        pipeline_cache = PipelineCache.getInstance()
        return pipeline_cache.get_val(key, self._inference_schema_name)


class PluginOutputInfo:
    """QACC Plugin Output Info class."""

    def __init__(self, plugin_config=None, mem_obj=None, status=1, out_dir=None, fname=None):
        # output path for the path and directory based plugins
        self.outdir = out_dir
        self.plugin_config = plugin_config
        self.mem_obj = mem_obj
        self.fname = fname
        self.status = status
        self.extn = '.raw'

        # Path plugins may call setPathOutput() to change the path.
        # Otherwise it may call getOutputPath() to get the desired path.

        # Dir plugins must need to call setDirOutputs() after the processing even if they
        # dont change any file names.

        # This is used only by plugins which update the file paths.
        self.updated_path = None

    def set_status(self, status):
        self.status = status

    def set_mem_output(self, memObj):
        assert (self.plugin_config._output_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_MEM)
        self.mem_obj = memObj

    def set_path_output(self, out_path):
        assert (self.plugin_config._output_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_PATH)
        self.updated_path = out_path

    def set_dir_outputs(self, out_path_list):
        assert (self.plugin_config._output_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_DIR)
        self.updated_path = out_path_list

    def is_memory_output(self):
        return self.plugin_config._output_info[qcc.IO_TYPE] == qcc.PLUG_INFO_TYPE_MEM

    def set_output_extn(self, extn):
        self.extn = extn

    def get_output_path(self):
        if self.updated_path:
            return self.updated_path
        return os.path.join(self.outdir, self.fname + self.extn)

    def get_output_dtype(self):
        return self.plugin_config._output_info[qcc.IO_DTYPE]

    def get_output_format(self):
        return self.plugin_config._output_info[qcc.IO_FORMAT]

    def get_out_dir(self):
        return self.outdir


class MetricInputInfo:

    def __init__(self, plugin_config, result_file, orig_inp_file, gt_file, out_dir):
        # Stores the entire plugin configuration
        self.plugin_config = plugin_config

        self.res_file = result_file
        self.orig_inp_file = orig_inp_file
        self.gt_file = gt_file

        # inference_schema key for pipeline pipeline_cache
        inference_schema = [s for s in out_dir.split('/') if s.startswith('schema')]
        self._inference_schema_name = inference_schema[0] if len(inference_schema) != 0 else None

    def get_param(self, param, default=None):
        if self.plugin_config.params is None:
            return default
        if param in self.plugin_config.params:
            return self.plugin_config.params[param]
        return default

    def get_groundtruth(self):
        return self.gt_file

    def get_result_file(self):
        return self.res_file

    def get_orig_inp_file(self):
        return self.orig_inp_file

    def get_inference_schema_name(self):
        return self._inference_schema_name[6:] if self._inference_schema_name is not None else None

    # API to access pipeline_cache
    def read_pipeline_cache_val(self, key):
        pipeline_cache = PipelineCache.getInstance()
        return pipeline_cache.get_val(key, self._inference_schema_name)


class MetricResult:

    def __init__(self, status=1, out_dir=None):
        self.status = status
        self.result_str = None
        self.result = {}
        self.out_dir = out_dir

    def get_out_dir(self):
        return self.out_dir

    def set_result_str(self, str):
        self.result_str = str

    def set_status(self, status):
        self.status = status

    def set_result(self, result_dict):
        self.result = result_dict
        if self.result_str is None:
            temp_str = ''
            for key, value in self.result.items():
                temp_str += f"{key}: {value}\n"
            self.set_result_str(temp_str)


class DatasetPluginInputInfo:

    def __init__(self, plugin, dataset_config):
        self._plugin_params = plugin.params
        self._dataset_config = dataset_config

    def get_base_path(self):
        return self._dataset_config.path

    def get_inputlist_file(self):
        return self._dataset_config.inputlist_file

    def get_annotation_file(self):
        return self._dataset_config.annotation_file

    def get_calibration_type(self):
        return self._dataset_config.calibration_type

    def get_calibration_file(self):
        return self._dataset_config.calibration_file

    def get_param(self, param, default=None):
        if self._plugin_params is None:
            return default
        if param in self._plugin_params:
            return self._plugin_params[param]
        return default


class DatasetPluginOutputInfo:

    def __init__(self, out_dir):
        self.out_dir = out_dir
        self.status = 1  # default status set to failure
        self.inputlist_file = None
        self.annotation_file = None
        self.calibration_type = None
        self.calibration_file = None
        self.inputlist_path_modified = False
        self.calibration_path_modified = False

    def get_out_dir(self):
        return self.out_dir

    def set_status(self, status):
        self.status = status

    def get_status(self):
        return self.status

    def set_inputlist_file(self, inputlist_file):
        self.inputlist_file = inputlist_file

    def set_annotation_file(self, annotation_file):
        self.annotation_file = annotation_file

    def set_calibration_type(self, calibration_type):
        self.calibration_type = calibration_type

    def set_calibration_file(self, calibration_file):
        self.calibration_file = calibration_file

    def set_inputlist_path_modified(self, inputlist_path_modified):
        self.inputlist_path_modified = inputlist_path_modified

    def set_calibration_path_modified(self, calibration_path_modified):
        self.calibration_path_modified = calibration_path_modified


class qacc_plugin:
    """QACC Plugin Interface."""

    def __init__(self):
        pass

    def execute(self, plugin_input_info_list: PluginInputInfo,
                plugin_output_info_list: PluginOutputInfo):
        """Execute method for the plugin class.

        Args:
            plugin_input_info: Takes an list of PluginInputInfo and PluginOutputInfo

        Returns:
        """
        pass


class qacc_metric:

    def execute(self, m_in: MetricInputInfo, m_out: MetricResult):
        """Execute method for the metric plugin class.

        Args:
            plugin_input_info: Takes an instance of MetricInputInfo and MetricResult

        Returns:
        """


class qacc_dataset:

    def execute(self, d_in: DatasetPluginInputInfo, d_out: DatasetPluginOutputInfo):
        """Execute method for dataset plugin class."""

