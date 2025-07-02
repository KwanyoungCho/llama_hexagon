# ==============================================================================
#
#  Copyright (c) 2020-2024 Qualcomm Technologies, Inc.
#  All Rights Reserved.
#  Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================
"""Prepared Model Definition Manager related classes and methods"""

import functools
import collections
import logging
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple, Set

# pylint: disable=import-error
from qti.aisw.converters.common import ir_graph as ir_graph_lib

_logger = logging.getLogger('TorchEmitter')
_class_name_counter = collections.defaultdict(int)
_block_name_to_class_name: Dict[str, str] = {}


def _get_import_codes(custom_module_paths: List[str]) -> List[str]:
    """
    Get import related code lines to be used in the definition of prepared model

    :return: Import related code lines
    """
    if custom_module_paths:
        file_paths = [Path(module_path).resolve() for module_path in custom_module_paths]
        custom_module_import_statements = "\n".join([f"from {file_path.stem} import *" for file_path in file_paths])

        custom_module_codes = [
            "\nfrom qti.aisw.emitter.utils.torch_utils import import_from_path\nfrom pathlib import PosixPath",
            "[",
            "\timport_from_path(file_path.stem, file_path)",
            f"\tfor file_path in {file_paths}",
            "]",
            custom_module_import_statements
        ]
    else:
        custom_module_codes = []
    return [
        "import torch",
        "import torchvision",
        "from qti.aisw.emitter import emitter_ops",
        "try:",
        "\timport aimet_torch.nn.modules.custom as elementwise_ops",
        "except ImportError:",
        "\tfrom qti.aisw.emitter import elementwise_ops",
    ] + custom_module_codes


def _extract_block_input_tensors(module_inputs: List[str],
                                 block_input_tensors: List[str],
                                 block_output_tensors: Set[str]) -> List[str]:
    """
    Find possible block inputs from module inputs

    :param module_inputs: Module input tensor names
    :param block_input_tensors: Block input tensor names
    :param block_output_tensors: Block output tensor names
    :return: Input tensor names to be included in block input tensors
    """
    input_tensors = []
    block_input_tensors = set(block_input_tensors)
    for module_input in module_inputs:
        if module_input in block_output_tensors:
            continue

        is_already_included = module_input in block_input_tensors
        is_temp_transposed_tensor = module_input.startswith("temp_transpose_")
        is_module_buffer = module_input.startswith("self.")

        if not is_already_included and not is_temp_transposed_tensor and not is_module_buffer:
            input_tensors.append(module_input)

    return input_tensors


def _extract_block_create_codes(module_create_codes: List[str], block_name: str) -> List[str]:
    """
    Extract block create codes from module create codes

    :param module_create_codes: Module create codes
    :param block_name: Block name
    :return: Init codes to be included in block init codes
    """
    block_create_codes = []

    for create_code in module_create_codes:
        prefix = f"\t\tself.{block_name}"
        if create_code.startswith(prefix):
            create_code = create_code.replace(prefix, "\t\tself")
        block_create_codes.append(create_code)

    return block_create_codes


def _extract_block_forward_pass_codes(module_forward_pass_codes: List[str],
                                      block_output_tensors: Set[str],
                                      block_name: str) -> Tuple[List[str], Set[str]]:
    """
    Extract block forward pass codes from module forward pass codes

    :param module_forward_pass_codes: Module forward pass codes
    :param block_output_tensors: Block output tensor names
    :param block_name: Block name
    :return: Forward pass codes to be included in block forward pass codes
        and released output tensor names to be excluded from block output tensors
    """
    forward_pass_codes = []
    released_output_tensors = set()

    for code in module_forward_pass_codes:
        lhs, rhs = code.split(" = ")
        rhs = rhs.replace(f"self.{block_name}", "self")
        forward_pass_codes.append(f"{lhs} = {rhs}")
        if rhs == "None" and lhs.lstrip("\t") in block_output_tensors:
            released_output_tensors.add(lhs.lstrip("\t"))

    return forward_pass_codes, released_output_tensors


class ModelDefinitionManager:
    """
    Model Definition Manager class (Flattened Model) to maintain model definition including create and execute codes
    """
    def __init__(self, custom_module_paths: Optional[List[str]] = None):
        self._module_definition_codes = []
        self._leaf_module_create_codes = []
        self._complex_module_create_codes = []
        self._model_input_codes = []
        self._execute_codes = []
        self._model_output_codes = []
        self._custom_module_paths = [] if custom_module_paths is None else custom_module_paths

    def add_module_definition_code(self, definition_code: str):
        """
        Add module definition related codes

        :param definition_code: Code required to define class
        """
        self._module_definition_codes.append(definition_code)

    # pylint: disable=unused-argument
    def add_leaf_module_create_code(self,
                                    create_code: str,
                                    op: Optional[ir_graph_lib.IrOp] = None,
                                    module_name: Optional[str] = None):
        """
        Add leaf module instantiation codes

        :param create_code: Leaf module instantiation code
        :param op: IrOp corresponding to create code
        :param module_name: Prepared module name corresponding to create code
        """
        self._leaf_module_create_codes.append(create_code)

    def add_complex_module_create_code(self, create_code: str):
        """
        Add complex (nested, module list) instantiation codes

        :param create_code: Complex module instantiation code
        """
        self._complex_module_create_codes.append(create_code)

    # pylint: disable=unused-argument
    def add_execute_code(self,
                         execute_code: str,
                         op: Optional[ir_graph_lib.IrOp] = None,
                         string_inputs: Optional[str] = None,
                         string_outputs: Optional[str] = None):
        """
        Add execute code in forward pass

        :param execute_code: Execute code run in forward pass
        :param op: IrOp corresponding to create code
        :param string_inputs: String representation of inputs
        :param string_outputs: String representation of outputs
        """
        self._execute_codes.append(execute_code)

    def add_model_input_code(self, model_input_code: str):
        """
        Add model input code

        :param model_input_code: Model input code
        """
        self._model_input_codes.append(model_input_code)

    def add_model_output_code(self, model_output_code: str):
        """
        Add model output code

        :param model_output_code: Model output code
        """
        self._model_output_codes.append(model_output_code)

    def get_model_definition(self) -> str:
        """
        Get complete PyTorch model definition

        :return: Executable PyTorch model definition string
        """
        import_block = "\n".join(_get_import_codes(self._custom_module_paths))
        init_block = self._get_init_block()
        forward_block = "\n".join(
            self._model_input_codes + self._execute_codes + self._model_output_codes
        )

        model_definition = "\n\n".join([import_block, init_block, forward_block])
        return model_definition

    def _get_init_block(self) -> str:
        """
        Get `__init__` block of model definition

        :return: init block of model definition
        """
        module_definition_block = "\n".join(self._module_definition_codes)
        complex_module_init_block = "\n".join(sorted(self._complex_module_create_codes))
        leaf_module_init_block = "\n".join(self._leaf_module_create_codes)

        return "\n".join(
            [module_definition_block, complex_module_init_block, leaf_module_init_block]
        )


class _PreparedModule:
    """
    Class to hold information of prepared torch module
    """
    def __init__(self,
                 create_codes: List[str] = None,
                 forward_pass_codes: List[str] = None,
                 name: str = None,
                 inputs: List[str] = None,
                 outputs: List[str] = None):
        self._create_codes = create_codes if create_codes else []
        self._forward_pass_codes = forward_pass_codes if forward_pass_codes else []
        self.name = name if name else None
        self.inputs = inputs if inputs else []
        self.outputs = outputs if outputs else []

    def add_create_code(self, create_code: str):
        """
        Add initialization related code

        :param create_code: Initialization related code
        """
        self._create_codes.append(create_code)

    def add_forward_pass_code(self, forward_pass_code: str):
        """
        Add forward pass related code

        :param forward_pass_code: Forward pass related code
        """
        self._forward_pass_codes.append(forward_pass_code)

    @property
    def forward_pass_codes(self) -> List[str]:
        """
        Property to get forward pass codes

        :return: Forward pass codes
        """
        return self._forward_pass_codes

    @property
    def create_codes(self) -> List[str]:
        """
        Property to get initialization codes

        :return: Initialization codes
        """
        return self._create_codes

    def get_module_definition(self, block_names_mapping) -> str:
        """
        Get module definition code

        :param block_names_mapping: Block to class name of mapping.
        :return: Module definition
        """
        class_name = block_names_mapping[self.name]
        class_index = _class_name_counter[class_name]
        _class_name_counter[class_name] += 1

        class_name = f"{class_name}{class_index}"
        _block_name_to_class_name[self.name] = class_name
        create_codes = [
            f"class {class_name}(torch.nn.Module):",
            "\tdef __init__(self):",
            "\t\tsuper().__init__()",
        ] + self.create_codes
        forward_pass_codes = (
            [f'\n\tdef forward(self, {", ".join(self.inputs)}):']
            + self.forward_pass_codes
            + [f"\t\treturn {', '.join(self.outputs)}"]
        )

        create_code_block = "\n".join(create_codes)
        forward_pass_code_block = "\n".join(forward_pass_codes)
        return "\n".join([create_code_block, forward_pass_code_block])


class StructuredModelDefinitionManager:
    """
    Model Definition Manager class (Structured Model) to maintain model definition including create and execute codes
    """

    def __init__(self, custom_module_paths: Optional[List[str]] = None):
        self._module_definition_codes = []
        self._complex_module_create_codes = []
        self._model_input_codes = []
        self._model_output_codes = []
        self._op_to_prepared_module = collections.defaultdict(_PreparedModule)
        self._ops_included_in_blocks = set()
        self._op_to_block_definition: Dict[Any, _PreparedModule] = {}
        self._complex_module_create_codes_to_skip = set()
        self._custom_module_paths = [] if custom_module_paths is None else custom_module_paths

    def add_module_definition_code(self, definition_code: str):
        """
        Add module definition related codes

        :param definition_code: Code required to define class
        """
        self._module_definition_codes.append(definition_code)

    def add_leaf_module_create_code(self,
                                    create_code: str,
                                    op: Optional[ir_graph_lib.IrOp] = None,
                                    module_name: Optional[str] = None):
        """
        Add leaf module instantiation codes

        :param create_code: Leaf module instantiation code
        :param op: IrOp corresponding to create code
        :param module_name: Prepared module name corresponding to create code
        """
        assert op is not None, "op must be provided in structured model definition manager"

        self._op_to_prepared_module[op].add_create_code(create_code)
        if module_name:
            self._op_to_prepared_module[op].name = module_name

    def add_complex_module_create_code(self, create_code: str):
        """
        Add complex (nested, module list) instantiation codes

        :param create_code: Complex module instantiation code
        """
        self._complex_module_create_codes.append(create_code)

    def add_model_input_code(self, model_input_code: str):
        """
        Add model input related code

        :param model_input_code: Model input code
        """
        self._model_input_codes.append(model_input_code)

    def add_model_output_code(self, model_output_code: str):
        """
        Add model output related code

        :param model_output_code: Model output code
        """
        self._model_output_codes.append(model_output_code)

    def add_execute_code(self,
                         execute_code: str,
                         op: Optional[ir_graph_lib.IrOp] = None,
                         string_inputs: Optional[str] = None,
                         string_outputs: Optional[str] = None):
        """
        Add execute code in forward pass

        :param execute_code: Execute code run in forward pass
        :param op: IrOp corresponding to create code
        :param string_inputs: String representation of inputs
        :param string_outputs: String representation of outputs
        """
        assert op is not None, "op must be provided in structured model definition manager"

        self._op_to_prepared_module[op].add_forward_pass_code(execute_code)

        if string_inputs and not self._op_to_prepared_module[op].inputs:
            self._op_to_prepared_module[op].inputs = [x.strip() for x in string_inputs.split(", ")]

        if string_outputs:
            self._op_to_prepared_module[op].outputs = [x.strip() for x in string_outputs.split(", ")]

    # pylint: disable=too-many-locals
    def get_structured_model_definition(self,
                                        model_name: str,
                                        block_names_mapping: Optional[List[str]]) -> str:
        """
        Get structured model definition to be written to file

        :param model_name: Prepared model class name
        :param original_model: Original torch model
        :param block_names_mapping: List of Block names along with the container class name
        :return: Structured model definition
        """
        refined_block_mapping = {re.sub(r"\.(\d+)", r"[\1]", block_name): v for block_name, v in block_names_mapping.items()}
        block_names = list(refined_block_mapping.keys())
        if block_names is None:
            block_names = []

        self._extract_intermediate_module_blocks(block_names)
        import_code_block = "\n".join(_get_import_codes(self._custom_module_paths))
        intermediate_class_codes = [block.get_module_definition(refined_block_mapping) for block in self._op_to_block_definition.values()]
        intermediate_class_code_block = "\n\n".join(intermediate_class_codes)
        complex_module_create_codes = [
            create_code
            for create_code in self._complex_module_create_codes
            if create_code not in self._complex_module_create_codes_to_skip
        ]

        forward_pass_codes = []
        leaf_module_create_codes = []
        for op, module in self._op_to_prepared_module.items():
            if op in self._ops_included_in_blocks:
                continue

            if block_definition := self._op_to_block_definition.get(op):
                complex_module_create_codes.append(
                    f"\t\tself.{block_definition.name} = {_block_name_to_class_name[block_definition.name]}()"
                )
                forward_pass_codes.append(
                    f"\t\t{', '.join(block_definition.outputs)} = self.{block_definition.name}({', '.join(block_definition.inputs)})"
                )
            else:
                leaf_module_create_codes.extend(module.create_codes)
                forward_pass_codes.extend(module.forward_pass_codes)

        complex_module_create_codes.sort()
        class_signature_codes = [f"class {model_name}(torch.nn.Module):", "\tdef __init__(self):", "\t\tsuper().__init__()"]
        init_code_block = "\n".join(class_signature_codes + complex_module_create_codes + leaf_module_create_codes)
        forward_pass_code_block = "\n".join(self._model_input_codes + forward_pass_codes + self._model_output_codes)
        return "\n\n".join([import_code_block, intermediate_class_code_block, init_code_block, forward_pass_code_block])

    # pylint: disable=too-many-locals
    def _extract_intermediate_module_blocks(self, block_names: List[str]):
        """
        Extract possible intermediate block and update self._op_to_block_definition dictionary

        :param block_names: Block names
        """

        for block_name in block_names:
            start_op, end_op = self._get_start_and_end_op(block_name)

            block_create_codes = []
            block_forward_pass_codes = []
            block_input_tensors = []
            block_output_tensors = set()
            for op, module in self._op_to_prepared_module.items():
                is_start_or_end_op = op in {start_op, end_op}
                is_descendant_of_block = module.name.startswith(block_name)
                is_input_from_previous_output = set(module.inputs).intersection(block_output_tensors)

                if is_start_or_end_op or is_descendant_of_block or is_input_from_previous_output:
                    # Extract block input tensors from module inputs and update
                    input_tensors = _extract_block_input_tensors(module.inputs, block_input_tensors, block_output_tensors)
                    block_input_tensors.extend(input_tensors)

                    # Modify module init codes to work on intermediate block
                    create_codes = _extract_block_create_codes(module.create_codes, block_name)
                    block_create_codes.extend(create_codes)

                    # Modify module forward pass codes to work on intermediate block
                    forward_pass_codes, released_output_tensors = _extract_block_forward_pass_codes(module.forward_pass_codes,
                                                                                                    block_output_tensors,
                                                                                                    block_name)
                    block_forward_pass_codes.extend(forward_pass_codes)

                    # Remove released output tensors in block forward pass codes and update module outputs
                    block_output_tensors.difference_update(released_output_tensors)
                    block_output_tensors.update(module.outputs)

                    if op == end_op:
                        break
                    self._ops_included_in_blocks.add(op)

            complex_module_create_codes = self._get_complex_module_create_codes_in_block(block_name)
            self._op_to_block_definition[end_op] = _PreparedModule(
                create_codes=complex_module_create_codes + block_create_codes,
                forward_pass_codes=block_forward_pass_codes,
                name=block_name,
                inputs=block_input_tensors,
                outputs=list(block_output_tensors),
            )

    def _get_start_and_end_op(self, block_name: str) -> Tuple[ir_graph_lib.IrOp, ir_graph_lib.IrOp]:
        """
        Find first and last occurred IrOp starting with block name

        :param block_name: Block name
        :return: Tuple of start and end IrOp
        """
        start_op = next((op for op, module in self._op_to_prepared_module.items() if module.name.startswith(block_name)))
        end_op = next((op for op, module in reversed(self._op_to_prepared_module.items()) if module.name.startswith(block_name)))

        return start_op, end_op

    def _get_complex_module_create_codes_in_block(self, block_name: str) -> List[str]:
        """
        Get complex module init codes that need to be in this intermediate block

        :param block_name: Block name
        :return: Complex module init codes to be in intermediate block
        """
        module_create_codes = []
        for create_code in self._complex_module_create_codes:
            prefix = f"\t\tself.{block_name}"
            lhs, _ = create_code.split(" = ")
            if lhs.startswith(prefix):
                self._complex_module_create_codes_to_skip.add(create_code)
                if lhs != prefix:
                    create_code = create_code.replace(prefix, "\t\tself")
                    module_create_codes.append(create_code)

        return sorted(module_create_codes)
