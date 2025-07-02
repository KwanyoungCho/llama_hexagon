# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
#  All rights reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import argparse
from . import io_utils
from qti.aisw.converters.common.utils.converter_utils import log_warning, log_info

valid_processor_choices = ('snapdragon_801', 'snapdragon_820', 'snapdragon_835')
valid_runtime_choices = ('cpu', 'gpu', 'dsp')

class ExportFormatType(object):
    """
    Contains supported export format types. This can be used during QAIRT to determine output DLC type
    """
    # DLC_DEFAULT (default):
    # - Produce a Float graph given a Float Source graph
    # - Produce a Quant graph given a Quant Source graph
    # DLC_STRIP_QUANT:
    # - Produce a Float Graph with discarding Quant data
    # DLC_FLOAT (hidden):
    # - Hidden option and is the legacy QAIRT behavior. If specified, output DLC will be having
    # Quantization Encodings cached in it (which can be applied during Quantization step)

    DLC_DEFAULT = "DLC_DEFAULT"
    DLC_STRIP_QUANT = "DLC_STRIP_QUANT"
    DLC_FLOAT = "DLC_FLOAT"

    @classmethod
    def get_supported_types(cls):
        return [cls.DLC_DEFAULT, cls.DLC_STRIP_QUANT, cls.DLC_FLOAT]


class ValidateTargetArgs(argparse.Action):
    def __call__(self, parser, args, values, option_string=None):
        specified_runtime, specified_processor = values
        if specified_runtime not in valid_runtime_choices:
            raise ValueError('invalid runtime_target {s1!r}. Valid values are {s2}'.format(s1=specified_runtime,
                                                                                           s2=valid_runtime_choices)
                             )
        if specified_processor not in valid_processor_choices:
            raise ValueError('invalid processor_target {s1!r}. Valid values are {s2}'.format(s1=specified_processor,
                                                                                             s2=valid_processor_choices)
                             )
        setattr(args, self.dest, values)


def check_filename_encoding(filename):
    try:
        filename.encode('utf-8')
    except UnicodeEncodeError:
        raise ValueError("Converter expects string arguments to be UTF-8 encoded: %s" % filename)


# Validation for generic file, optional validation for file existing already
def validate_filename_arg(*, must_exist=False, create_missing_directory=False, is_directory=False):
    class ValidateFilenameArg(argparse.Action):
        def __call__(self, parser, args, value, option_string=None):
            check_filename_encoding(value)
            io_utils.check_validity(value, create_missing_directory=create_missing_directory, must_exist=must_exist, is_directory=is_directory)
            setattr(args, self.dest, value)

    return ValidateFilenameArg


# Validation for the path of generic file or folder
def validate_pathname_arg(*, must_exist=False):
    class ValidatePathnameArg(argparse.Action):
        def __call__(self, parser, args, value, option_string=None):
            check_filename_encoding(value)
            io_utils.check_validity(value, is_path=True, must_exist=must_exist)
            setattr(args, self.dest, value)

    return ValidatePathnameArg


def check_xml():
    class ValidateXmlFileArgs(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            for value in values:
                io_utils.check_validity(value, extensions=[".xml"])
            if hasattr(args, self.dest) and getattr(args, self.dest) is not None:
                old_values = getattr(args, self.dest)
                values.extend(old_values)
            setattr(args, self.dest, values)

    return ValidateXmlFileArgs

def check_json():
    class ValidateXmlFileArgs(argparse.Action):
        def __call__(self, parser, args, values, option_string=None):
            for value in values:
                io_utils.check_validity(value, extensions=[".json"])
            if hasattr(args, self.dest) and getattr(args, self.dest) is not None:
                old_values = getattr(args, self.dest)
                values.extend(old_values)
            setattr(args, self.dest, values)
    return ValidateXmlFileArgs

def validate_export_format_option():
    class ValidateExportFormatArgs(argparse.Action):
        def __call__(self, parser, args, value, option_string=None):
            if value not in ExportFormatType.get_supported_types():
                raise ValueError("Invalid --export_format option provided. Valid options are 'DLC_DEFAULT', 'DLC_STRIP_QUANT'")
            setattr(args, self.dest, value)
    return ValidateExportFormatArgs

def two_hex(hex_pair):
    hex_numbers = hex_pair.split()
    if len(hex_numbers) != 2:
        raise argparse.ArgumentError
    try:
        values = list(map(lambda x: int(x, 16), hex_numbers))
    except Exception as e:
        raise Exception("{}: Could not represent quantization step argument: {} as a valid integer"
                        .format(str(e), hex_numbers))
    return values

def validate_tensor_names_in_graph(tensor_names, py_graph, tensor_names_source_file, skip_validation=False):
    tensors_not_in_graph = [name for name in tensor_names if not py_graph.has_buffer(name)]

    if tensors_not_in_graph:
        if skip_validation:
            message = "The following tensors from {} were not found in the graph. Proceeding as --skip_validation is requested.\n{}" \
                .format(tensor_names_source_file, "\n".join(sorted(tensors_not_in_graph)))
            log_warning(message)
        else:
            message = "The following tensors from {} were not found in the graph.\n" \
                .format(tensor_names_source_file, "\n".join(sorted(tensors_not_in_graph)))
            raise RuntimeError(message)
    else:
        log_info("All tensors from {} were found in the graph.".format(tensor_names_source_file))
