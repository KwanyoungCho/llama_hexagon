# =============================================================================
#
#  Copyright (c) Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

QNN_FLOAT_16 = '0x216'
QNN_FLOAT_32 = '0x232'
QNN_SFIXED_4 = '0x304'
QNN_SFIXED_8 = '0x308'
QNN_SFIXED_16 = '0x316'
QNN_UFIXED_4 = '0x404'
QNN_UFIXED_8 = '0x408'
QNN_UFIXED_16 = '0x416'

QNN_PARAM_TYPE = 4

def convert_qnn_enc_to_qairt(qnn_encoding: dict) -> dict:
    '''
    Given qnn encodings, convert it to qairt format encodings
    :param qnn_encoding: qnn encoding for a tensor
    :return: qairt_encoding for the tensor
    '''

    def _format_qnn_enc(encoding: dict, data_type: int) -> dict:
        '''
        format the qnn encoding to qairt format
        :param encoding: qnn encoding
        '''
        data_type = hex(data_type)
        if data_type == QNN_FLOAT_16:
            return {'dtype': 'float', 'bitwidth': 16}
        elif data_type == QNN_FLOAT_32:
            return {'dtype': 'float', 'bitwidth': 32}
        elif data_type in [QNN_UFIXED_4, QNN_UFIXED_8, QNN_UFIXED_16, QNN_SFIXED_4, QNN_SFIXED_8, QNN_SFIXED_16]:
            _qairt_encoding = {}
            _qairt_encoding['bitwidth'] = encoding['bitwidth']
            _qairt_encoding['is_symmetric'] = str(encoding['is_symmetric'])
            _qairt_encoding['max'] = encoding['maximum']
            _qairt_encoding['min'] = encoding['minimum']
            _qairt_encoding['offset'] = encoding['offset']
            _qairt_encoding['scale'] = encoding['scale']

            return _qairt_encoding
        else:
            return {}

    quant_params = qnn_encoding['quant_params']
    if 'scale_offset' in quant_params:
        formated_enc = _format_qnn_enc(quant_params['scale_offset'], qnn_encoding['data_type'])
        qairt_encoding = [formated_enc] if formated_enc else []
    else:
        # per channel quant encodings
        qairt_encoding = []
        for enc in quant_params['axis_scale_offset']['scale_offsets']:
            formated_enc = _format_qnn_enc(enc, qnn_encoding['data_type'])
            if formated_enc:
                qairt_encoding.append(formated_enc)
            else:
                return []

    return qairt_encoding


def add_encodings(encodings: dict, tensor_info: dict, version: str) -> dict:
    '''
    Adds tensor_encodings given in tensor_info to encodings dictionary inplace

    :param encodings: encodings data structure to which tensor_info encodings
        needs to be added
    :param tensor_info: a dictionary data structure as follow:
        {
            'name': name of the tensor
            'encoding': encoding value
            'type': one of (activation_encodings, param_encodings). str value
        }
    :param version: encodings data_structure version
    '''

    tensor_name = tensor_info['name']
    tensor_type = tensor_info['type']
    tensor_encoding = tensor_info['encoding']

    if version == 'legacy':
        encodings[tensor_type][tensor_name] = tensor_encoding
    elif version == '1.0.0':
        # Delte the enodings for tenosr_name if present
        for idx, enc in enumerate(encodings[tensor_type]):
            if enc['name'] == tensor_name:
                del encodings[tensor_type][idx]
                break
        encodings[tensor_type].append(tensor_encoding)

    return encodings


def get_encodings_structure(version: str) -> dict:
    '''
    Given encodings version returns empty encodings data structure

    :param version: version value
    :return encodings_struct:
    '''
    if version == 'legacy':
        return {'activation_encodings': {}, 'param_encodings': {}}
    elif version == '1.0.0':
        return {'version': version, 'activation_encodings': [], 'param_encodings': []}
    else:
        return None


def get_resolved_names(tensor_name: str) -> list:
    '''
    :return: list of resolved tensor names
    :param tensor_name: QNN tensor name
    '''

    # TODO: use framework_op_trace to resolve the target name
    # once the feature is stable

    resolved_names = []
    if '_' in tensor_name:
        resolved_names.append('_'.join(tensor_name.split('_')[:-1]))
    if '.' in tensor_name:
        resolved_names.append('.'.join(tensor_name.split('.')[:-1]))

    return resolved_names


def get_dtype(enc: dict) -> str:
    '''
    returns the dtype for the tensor given its encodings profile

    :param enc: encodings dict for a tensor
    '''
    if 'dtype' in enc[0] and enc[0]['dtype'] == "float":
        return "float"
    if 'scale' in enc[0] and enc[0]['scale'] != 0:
        return 'int'
    return 'float'


def needs_encoding_update(enc1: dict | list, enc2: dict | list, version: str) -> bool:
    '''
    returns whether enc1 should overwrites enc2 with precedence
    Precendence: int16>int8>int4>fp32>fp16>fp8

    :param enc1: encodings dict for a tensor
    :param enc2: encodings dict for a tensor
    :param version: version value
    :raise Exception: if version not in (v1, v2)
    :return: True, if encoding enc1 must override enc2 else False
    '''
    if version == 'legacy':
        dtype1 = get_dtype(enc1)
        dtype2 = get_dtype(enc2)
        bitwidth1 = enc1[0]['bitwidth']
        bitwidth2 = enc2[0]['bitwidth']
    elif version == '1.0.0':
        dtype1 = enc1['dtype'].lower()
        dtype2 = enc2['dtype'].lower()
        bitwidth1 = enc1['bw']
        bitwidth2 = enc2['bw']
    else:
        raise Exception(f"Encodings version {version} not supported")

    dtype_order = {'int': [4, 8, 16], 'float': [8, 16, 32]}
    if dtype1 == dtype2:
        return dtype_order[dtype1].index(bitwidth1) > dtype_order[dtype2].index(bitwidth2)

    # for precedence: int > float
    return dtype1 == 'int'


def identify_inter_activations_path(current_activation: str, parent_activation_name: str,
                                    target_activation_op_map: dict, depth: int) -> list:
    '''
    returns the path between children op activation and target op activation
    in the target graph

    :param current_activation: children op activation in target graph
    :param parent_activation_name: parent op activation in target graph
    :param target_activation_op_map: target activation as key and target op as value
    :param depth: current number of ops in the path between parent and children ops.
    If > 10, the path is dropped as it may indicates loops.
    '''
    # Base Case
    if current_activation == parent_activation_name:
        return [parent_activation_name]
    if depth == 10:
        return []

    smallest_path = []

    if current_activation in target_activation_op_map:
        current_target_op = target_activation_op_map[current_activation]
        for input in current_target_op.get_inputs():
            path = identify_inter_activations_path(input, parent_activation_name,
                                                   target_activation_op_map, depth + 1)
            #         |------------------>|
            # 100 --->|                   |-----> 103
            #         |--> 101 --> 102 -->|
            # Incase of residual connections, path between 100 and 103
            # should be {100, 103} but one other possible path is
            # {100, 101, 102, 103}. Therefore, we need to take the smallest
            # path
            if path:
                if not smallest_path:
                    smallest_path = path
                else:
                    smallest_path = smallest_path if len(smallest_path) < len(path) else path

    # Add current activation to path if path not empty
    if smallest_path:
        smallest_path.append(current_activation)

    return smallest_path


def is_convert_op_in_path(path: list, target_activation_op_map: dict) -> tuple:
    '''
    return True, convert_activation if there exists convert op in the path
    else False

    :param path: list of target activation
    :param target_activation_op_map: target activation as key and target op as value
    '''
    for activation in path:
        if activation in target_activation_op_map:
            if 'converted_QNN_DATATYPE' in activation:
                return True, activation
    return False, None


def get_framework_type(model_path: str) -> str:
    '''
    returns the framework type given the model path using the extension
    '''
    framework_type = model_path.split('.')[-1]
    if framework_type.lower() == "onnx":
        return 'onnx'
    elif framework_type.lower() == "pb":
        return 'tensorflow'
    else:
        raise Exception(f"framework type {framework_type} not supported.")
