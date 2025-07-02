# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================


class Constants:
    # Plugin constants.
    PLUG_INFO_TYPE_MEM = 'mem'
    PLUG_INFO_TYPE_PATH = 'path'
    PLUG_INFO_TYPE_DIR = 'dir'

    # IO info keys
    IO_TYPE = 'type'
    IO_DTYPE = 'dtype'
    IO_FORMAT = 'format'

    # Datatypes
    DTYPE_FLOAT16 = 'float16'
    DTYPE_FLOAT32 = 'float32'
    DTYPE_FLOAT64 = 'float64'
    DTYPE_INT8 = 'int8'
    DTYPE_INT16 = 'int16'
    DTYPE_INT32 = 'int32'
    DTYPE_INT64 = 'int64'

    # Formats
    FMT_NPY = 'np'
    FMT_CV2 = 'cv2'
    FMT_PIL = 'pil'

    # Plugin status
    STATUS_SUCCESS = 0
    STATUS_ERROR = 1
    STATUS_REMOVE = 2

    # Pipeline stage names
    STAGE_PREPROC_CALIB = 'calibration'
    STAGE_PREPROC = 'preproc'
    STAGE_POSTPROC = 'postproc'
    STAGE_COMPILE = 'compiled'
    STAGE_INFER = 'infer'
    STAGE_METRIC = 'metric'

    # File type
    BINARY_PATH = 'binary'

    # output file names
    PROCESSED_OUTFILE = 'processed-outputs.txt'
    QNN_PROCESSED_OUTFILE = 'qnn-processed-outputs.txt'
    INFER_OUTFILE = 'infer-outlist.txt'
    PROFILE_YAML = 'profile.yaml'
    RESULTS_TABLE_CSV = 'metrics-info.csv'
    PROFILING_TABLE_CSV = 'profiling-info.csv'
    INPUT_LIST_FILE = 'processed-inputlist.txt'
    CALIB_FILE = 'processed-calibration.txt'
    INFER_RESULTS_FILE = 'runlog_inf.txt'

    # output directory names
    DATASET_DIR = 'dataset'

    # qacc inference schema runstatus
    SCHEMA_INFER_SUCCESS = 0
    SCHEMA_INFER_FAIL = 1
    SCHEMA_POSTPROC_SUCCESS = 2
    SCHEMA_POSTPROC_FAIL = 3
    SCHEMA_METRIC_SUCCESS = 4
    SCHEMA_METRIC_FAIL = 5
    SCHEMA_COMPARATOR_FAIL = 6
    SCHEMA_COMPARATOR_SUCCESS = 7
    SCHEMA_PREPROC_SUCCESS = 8
    SCHEMA_PREPROC_FAIL = 9
    SCHEMA_COMPILE_FAIL = 10
    SCHEMA_COMPILE_SUCCESS = 11

    def get_inference_schema_status(code):
        if code == Constants.SCHEMA_INFER_FAIL:
            return 'Inference Failed'
        elif code == Constants.SCHEMA_POSTPROC_FAIL:
            return 'PostProcess Failed'
        elif code == Constants.SCHEMA_METRIC_FAIL:
            return 'Metric Failed'
        elif code == Constants.SCHEMA_COMPARATOR_FAIL:
            return 'Comparator Failed'
        elif code == Constants.SCHEMA_PREPROC_FAIL:
            return 'Preprocess Failed'
        elif code == Constants.SCHEMA_COMPILE_FAIL:
            return 'Compilation Failed'
        else:
            return 'Success'

    # search space delimiter
    SEARCH_SPACE_DELIMITER = '|'
    RANGE_BASED_DELIMITER = '-'
    RANGE_BASED_SWEEP_PREFIX = 'range=('

    # cleanup options
    CLEANUP_AT_END = 'end'
    CLEANUP_INTERMEDIATE = 'intermediate'
    INFER_SKIP_CLEANUP = '/temp/'

    # config info
    MODEL_INFO_BATCH_SIZE = 'batchsize'

    # TODO: Remove Pipeline Cache related items
    # pipeline pipeline_cache keys
    PIPELINE_BATCH_SIZE = 'config.info.batchsize'
    PIPELINE_WORK_DIR = 'qacc.work_dir'
    PIPELINE_MAX_INPUTS = 'qacc.dataset.max_inputs'
    PIPELINE_MAX_CALIB = 'qacc.dataset.max_calib'

    # preproc
    PIPELINE_PREPROC_DIR = 'qacc.preproc_dir'
    PIPELINE_PREPROC_FILE = 'qacc.preproc_file'
    # calib
    PIPELINE_CALIB_DIR = 'qacc.calib_dir'
    PIPELINE_CALIB_FILE = 'qacc.calib_file'
    PIPELINE_PREPROC_IS_CALIB = 'qacc.preproc_is_calib'
    # postproc
    PIPELINE_POSTPROC_DIR = 'qacc.postproc_dir'  # contains nested structure
    PIPELINE_POSTPROC_FILE = 'qacc.postproc_file'  # contains nested structure
    # infer
    PIPELINE_INFER_DIR = 'qacc.infer_dir'  # contains nested structure
    PIPELINE_INFER_FILE = 'qacc.infer_file'  # contains nested structure
    PIPELINE_INFER_INPUT_INFO = 'qacc.infer_input_info'
    PIPELINE_INFER_OUTPUT_INFO = 'qacc.infer_output_info'
    PIPELINE_NETWORK_BIN_DIR = 'qacc.network_bin_dir'  # contains nested structure
    PIPELINE_NETWORK_DESC = 'qacc.network_desc'  # contains nested structure
    PIPELINE_PROGRAM_QPC = 'qacc.program_qpc'  # contains nested structure

    # internal pipeline cache keys
    INTERNAL_CALIB_TIME = 'qacc.calib_time'
    INTERNAL_PREPROC_TIME = 'qacc.preproc_time'
    INTERNAL_QUANTIZATION_TIME = 'qacc.quantization_time'  # contains nested structure
    INTERNAL_COMPILATION_TIME = 'qacc.compilation_time'  # contains nested structure
    INTERNAL_INFER_TIME = 'qacc.infer_time'  # contains nested structure
    INTERNAL_POSTPROC_TIME = 'qacc.postproc_time'  # contains nested structure
    INTERNAL_METRIC_TIME = 'qacc.metric_time'  # contains nested structure
    INTERNAL_EXEC_BATCH_SIZE = 'qacc.exec_batch_size'

    # file naming convention
    NETWORK_DESC_FILE = 'networkdesc.bin'
    PROGRAM_QPC_FILE = 'programqpc.bin'

    # options for get orig file paths API
    LAST_BATCH_TRUNCATE = 1
    LAST_BATCH_REPEAT_LAST_RECORD = 2
    LAST_BATCH_NO_CHANGE = 3

    # dataset filter plugin keys
    DATASET_FILTER_PLUGIN_NAME = 'filter_dataset'
    DATASET_FILTER_PLUGIN_PARAM_RANDOM = 'random'
    DATASET_FILTER_PLUGIN_PARAM_MAX_INPUTS = 'max_inputs'
    DATASET_FILTER_PLUGIN_PARAM_MAX_CALIB = 'max_calib'

    # QNN related cache keys
    QNN_SDK_DIR = "qnn_sdk_dir"

    DEFAULT_ADB_PATH = "/opt/bin/adb"
    DEFAULT_MODEL_ZOO_PATH = "/home/model_zoo"
    CONTEXT_BACKEND_EXTENSION_CONFIG = "context_backend_extensions.json"
    NETRUN_BACKEND_EXTENSION_CONFIG = "netrun_backend_extensions.json"

    PIPE_SUPPORTED_QUANTIZER_PARAMS = {
        "param_quantizer_calibration": "",
        "param_quantizer_schema": "",
        "act_quantizer_calibration": "",
        "act_quantizer_schema": "",
        "float_bitwidth": "float",
        "bias_bitwidth": "bias",
        "act_bitwidth": "act",
        "weights_bitwidth": "weight",
        "float_bias_bitwidth": "floatbias",
        "use_per_channel_quantization": "pcq",
        "use_per_row_quantization": "prq",
        "percentile_calibration_value": "pcv"
    }
    CUSTOM_OP_FLAGS = ['op_package_config', 'op_package_lib', 'package_name']
    SIMPLIFIED_CLEANED_MODEL_ONNX = 'cleanmodel_simplified.onnx'
    CLEANED_MODEL_ONNX = 'cleanmodel.onnx'
    CLEANED_MODEL_PT = 'cleanmodel.pt'
