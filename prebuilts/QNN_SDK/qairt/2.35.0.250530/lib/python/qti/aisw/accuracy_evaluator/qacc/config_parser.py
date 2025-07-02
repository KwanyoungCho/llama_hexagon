# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

from typing import Dict, List
import yaml
import os
import numpy as np

from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
from qti.aisw.accuracy_evaluator.common.utilities import Helper
from qti.aisw.accuracy_evaluator.qacc.utils import convert_npi_to_json, cleanup_quantization_overrides


class ConfigurationException(Exception):
    """Exceptions encountered in Configuration class."""

    def __init__(self, msg, exc=None):
        self.exc = exc
        self.msg = msg
        super().__init__(self.msg)

    def __str__(self):
        if self.exc == None:
            return 'Reason: {}'.format(self.msg)
        else:
            return 'Exc: {} Reason: {}'.format(self.exc, self.msg)


class ParserHelper:

    @classmethod
    def read_yaml_and_replace_globals(cls, config_path, set_global):
        with open(config_path, 'r') as stream:
            try:
                config = yaml.safe_load(stream)
            except yaml.YAMLError as exc:
                raise ConfigurationException('incorrect configuration file', exc)
        with open(config_path, 'r') as stream:
            file_data = stream.read()
        if 'globals' in config['model'] and config['model']['globals']:
            gconfig = config['model']['globals']

            if len(gconfig) > 0:
                if set_global:
                    gconfig.update(set_global)
                for k, v in gconfig.items():
                    file_data = file_data.replace('$' + k, str(v))

        return file_data

    @classmethod
    def parse_plugins_section(cls, plugins_list) -> List:
        new_plugins_list = []
        for plugin in plugins_list:
            new_plugins_list.append(plugin['plugin'])
        return new_plugins_list

    @classmethod
    def parse_globals_section(cls, config) -> Dict:
        new_config = {'globals_dict': config}
        return new_config

    @classmethod
    def parse_dataset_section(cls, config) -> Dict:
        new_config = {}
        for k, v in config.items():
            if k == 'inputlist_file' or k == 'annotation_file':
                new_config[k] = os.path.join(config['path'], v)
            elif k == 'calibration':
                new_config['calibration_file'] = os.path.join(config['path'], v['file'])
                new_config['calibration_type'] = v['type']
            elif k == 'transformations':
                new_config['dataset_plugin_list'] = cls.parse_plugins_section(v)
            else:
                new_config[k] = v
        return new_config

    @classmethod
    def format_model_node_info(cls, configured_info, batchsize):
        fmt_info = None
        if configured_info:
            fmt_info = {}
            for out_info in configured_info:
                for k, m in out_info.items():
                    assert len(m.values(
                    )) == 2, 'Invalid format for input/output info. Should have type and shape keys'
                    if isinstance(m['shape'], list):
                        node_name = Helper.sanitize_node_names(str(k))
                        bs_inx = None
                        if '*' in m['shape']:
                            bs_inx = m['shape'].index('*')
                            m['shape'][bs_inx] = batchsize
                        fmt_info[node_name] = [m['type'], m['shape'], bs_inx]
                        # if node_type == 'output' and 'comparator' in m:
                        #     fmt_info[node_name].append(m['comparator'])
                    else:
                        raise ConfigurationException('Invalid shape in input/output info :{}.'
                                                     ' usage e.g: [1,224,224,3]'.format(m['shape']))
        return fmt_info

    @classmethod
    def parse_inference_engine_section(cls, config, batchsize, calib_file) -> Dict:
        new_config = {}
        model_path = None
        for k, v in config.items():
            if k == 'model_path':
                if not os.path.exists(v):
                    model_zoo_path = os.environ.get('MODEL_ZOO_PATH', qcc.DEFAULT_MODEL_ZOO_PATH)
                    model_path_relative_model_zoo = os.path.join(model_zoo_path, v)
                    if not os.path.exists(model_path_relative_model_zoo):
                        raise ConfigurationException(
                            f'Model path {model_path_relative_model_zoo} does not exist')
                    new_config[k] = model_path_relative_model_zoo
                else:
                    new_config[k] = v
                model_path = new_config[k]
            elif k == 'inference_schemas' and len(v) > 0:
                inf_schemas = []
                for idx, inference_schema in enumerate(v):
                    inference_schema['inference_schema'].update({'idx': idx})
                    if 'tag' in inference_schema['inference_schema']:
                        tags = inference_schema['inference_schema']['tag']
                        inference_schema['inference_schema']['tag'] = [
                            t.strip() for t in tags.split(',')
                        ]
                    if 'backend' in inference_schema['inference_schema']:
                        inference_schema['inference_schema']['backend'] = inference_schema[
                            'inference_schema']['backend'].upper()
                    if 'converter_params' in inference_schema['inference_schema']:
                        for ck, cv in inference_schema['inference_schema'][
                                'converter_params'].items():
                            if ck == 'quantization_overrides':
                                quant_overrides_file = cv
                                if not os.path.exists(quant_overrides_file):
                                    raise ConfigurationException(
                                        f'Quantization overrides file {quant_overrides_file} does not exist'
                                    )
                                fname, extn = os.path.splitext(quant_overrides_file)
                                if extn == '.yaml':
                                    converted_json_path = fname + '.json'
                                    convert_npi_to_json(npi_yaml_file=quant_overrides_file,
                                                        output_json=converted_json_path)
                                    inference_schema['inference_schema']['converter_params'][
                                        'quantization_overrides'] = converted_json_path
                                output_json_cleaned = f'{fname}_cleaned.json'
                                output_json_cleaned = cleanup_quantization_overrides(
                                    inference_schema['inference_schema']['converter_params']
                                    ['quantization_overrides'], model_path=model_path,
                                    outpath=output_json_cleaned)
                                inference_schema['inference_schema']['converter_params'][
                                    'quantization_overrides'] = output_json_cleaned
                            if ck == "preserve_io_datatype":
                                preserve_io_val = cv
                                if isinstance(preserve_io_val, str):
                                    inference_schema["inference_schema"]["converter_params"][
                                        "preserve_io_datatype"] = [
                                            p.strip() for p in preserve_io_val.split(",")
                                        ]
                                elif (isinstance(preserve_io_val, bool) and preserve_io_val):
                                    inference_schema["inference_schema"]["converter_params"][
                                        "preserve_io_datatype"] = "all"
                    if 'quantizer_params' in inference_schema['inference_schema']:
                        if 'input_list' not in inference_schema['inference_schema'][
                                'quantizer_params']:
                            inference_schema['inference_schema']['quantizer_params'][
                                'input_list'] = calib_file
                        for qk, qv in inference_schema['inference_schema'][
                                'quantizer_params'].items():
                            if qk in qcc.PIPE_SUPPORTED_QUANTIZER_PARAMS:
                                param_list = [
                                    v.strip() for v in str(qv).split(qcc.SEARCH_SPACE_DELIMITER)
                                ]
                                if qk not in qcc.PIPE_SUPPORTED_QUANTIZER_PARAMS and len(
                                        param_list) > 1:
                                    raise ConfigurationException(
                                        f"Pipe option not available for quantizer param {qk}")
                                new_param_list = []
                                for param in param_list:
                                    if str(param).startswith(qcc.RANGE_BASED_SWEEP_PREFIX) and \
                                            str(param).endswith(')'):
                                        try:
                                            start, end, step = param[
                                                len(qcc.RANGE_BASED_SWEEP_PREFIX):-1].split(
                                                    qcc.RANGE_BASED_DELIMITER)
                                            start, end, step = start.strip(), end.strip(
                                            ), step.strip()
                                            val_precision = max([
                                                len(start.split('.')[-1]),
                                                len(end.split('.')[-1]),
                                                len(step.split('.')[-1])
                                            ])
                                        except:
                                            raise ConfigurationException(
                                                "Check range based parameter syntax in"
                                                "inference_schema params in config file")
                                        _, start = Helper.get_param_dtype(start, return_val=True)
                                        _, end = Helper.get_param_dtype(end, return_val=True)
                                        _, step = Helper.get_param_dtype(step, return_val=True)
                                        end += (step / 2)  # to include 'end' in the range
                                        range_values = [
                                            float(f'{range_val:0.{val_precision}f}')
                                            for range_val in np.arange(start, end, step)
                                        ]
                                        new_param_list.extend(range_values)
                                    else:
                                        dtype, val = Helper.get_param_dtype(param, return_val=True)
                                        new_param_list.append(dtype(val))

                                inference_schema["inference_schema"]["quantizer_params"][
                                    qk] = new_param_list

                            # values of 'algorithms' param are expected to be in a list
                            if qk == "algorithms":
                                inference_schema["inference_schema"]["quantizer_params"][
                                    "algorithms"] = [a.strip() for a in str(qv).split(",")]

                            if qk == "preserve_io_datatype":
                                preserve_io_val = qv
                                if isinstance(preserve_io_val, str):
                                    inference_schema["inference_schema"]["quantizer_params"][
                                        "preserve_io_datatype"] = [
                                            p.strip() for p in preserve_io_val.split(",")
                                        ]
                                elif (isinstance(preserve_io_val, bool) and preserve_io_val):
                                    inference_schema["inference_schema"]["quantizer_params"][
                                        "preserve_io_datatype"] = "all"

                    inf_schemas.append(inference_schema['inference_schema'])
                new_config[k] = inf_schemas
            elif k == 'inputs_info' or k == 'outputs_info':
                new_config[k] = cls.format_model_node_info(v, batchsize)
                if k == 'inputs_info':
                    input_names = []
                    for inp_info_dict in v:
                        input_names.extend(list(inp_info_dict.keys()))
                    new_config['input_names'] = input_names
            elif k == 'device_ids':
                new_config[k] = v.split(',')
            else:
                new_config[k] = v
        return new_config

    @classmethod
    def parse_preprocessing_section(cls, config) -> Dict:
        new_config = {
            'preprocessing_plugin_list': cls.parse_plugins_section(config['transformations'])
        }
        return new_config

    @classmethod
    def parse_postprocessing_section(cls, config) -> Dict:
        new_config = {}
        for k, v in config.items():
            if k == 'transformations':
                new_config['postprocessing_plugin_list'] = cls.parse_plugins_section(v)
            else:
                new_config[k] = v
        return new_config

    @classmethod
    def parse_metrics_section(cls, config) -> dict:
        new_config = {'metrics_plugin_list': cls.parse_plugins_section(config['transformations'])}
        return new_config

    @classmethod
    def parse_verifier_section(cls, config: dict) -> dict:
        """Parse the given dictionary to create a dict adhering to the VerifierConfiguration
        Args:
            config: Dictionary of verifier params
        Returns:
            new_config: New dictionary in VerifierConfiguration format
        """
        new_config = {}
        for k, v in config.items():
            if k == 'type':
                comp_type = v.replace('_', ' ')
                new_config['type'] = comp_type
            else:
                new_config[k] = v
        return new_config
