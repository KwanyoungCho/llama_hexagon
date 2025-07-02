# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from collections import deque
import math
import os
import tempfile
from typing import Optional, Union
import onnx_graphsurgeon as gs
import onnx
import onnxruntime
from pydantic.functional_serializers import model_serializer
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.mha2sha.defs.print_colors import Colors
import rich
import numpy as np

from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.utils.pretty_print import PrettyPrintConstants, bold_text, create_rich_table
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.utils.encodings import AimetEncodings
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.onnx_model_helper import OnnxModelHelper
from qti.aisw.tools.core.utilities.qairt_logging import LogAreas, QAIRTLogger
from qti.aisw.tools.core.utilities.framework.frameworks.onnx.model_transformations.mha2sha.utils import onnx as ou


splitter_log_area = LogAreas.register_log_area("onnx_splitter")


def generate_random_inputs(model: onnx.ModelProto):

    def get_tensor_proto_shape(tp: onnx.ValueInfoProto):
        return [dim.dim_value for dim in tp.type.tensor_type.shape.dim]

    def get_tensor_proto_dtype(tp: onnx.ValueInfoProto):
        return onnx.helper.tensor_dtype_to_np_dtype(tp.type.tensor_type.elem_type)

    inputs = {}
    for inp in model.graph.input:
        shape = get_tensor_proto_shape(inp)
        dtype = get_tensor_proto_dtype(inp)
        if inp.name == "input_ids":
            inputs["input_ids"] = np.random.randint(1, 500, shape).astype(dtype)
        elif inp.name == "lora_alpha":
            inputs["lora_alpha"] = np.zeros(shape).astype(dtype)
        else:
            inputs[inp.name] = np.random.rand(*shape).astype(dtype)
    return inputs


def split_onnx(
    model: Union[str, onnx.ModelProto],
    encodings: Optional[dict],
    *,
    num_splits: int,
    split_embedding: bool = False,
    split_lm_head: bool = False,
    skip_verification: bool = False,
    log_level: str = "info"
) -> Union[list[str], list[onnx.ModelProto]]:
    """
    Splits the given ONNX model into multiple sub-models.

    This function splits the model in-place into the specified number of sub-models.
    It supports splitting embeddings and language model heads.

    Args:
        model (onnx.ModelProto): The ONNX model to be split.
        encodings (Optional[dict]): Aimet encodings for the model.
    Keyword-only Args:
        num_splits (int): The number of splits to be made. Default is 1.
        split_embedding (bool): If True, splits the embeddings. Default is False.
        split_lm_head (bool): If True, splits the language model head. Default is False.
        log_level (str): The logging level to be used during the splitting. Default is "info".

    Model Topology:
               │ ←─────────  layers[0]  ────────────→ │       │ ←─────────  layers[-1]  ───────────-----─→ │
               │                                      │       │                                            │
    embed  ────┬─────────── add 0 ─┬────────── add 1 ──  ┄┄ ┄─┬─────────────── add(n-2) ─┬──────────── add(n-1) ─── lmhead
             ↑ └─ norm ─ attn ─┘   └─ norm ─ ffn ─┘   ↑       ↑ └─ norm ─ attn ─┘        └─ norm ─ ffn ─┘  ↑
             │                                        │       │                                            │
             │                                        │       │                                            │
            valid splitting points
    """

    logger = QAIRTLogger.register_area_logger(splitter_log_area,
                                              level=log_level,
                                              formatter_val="extended",
                                              handler_list=["dev_console"])

    is_model_path = isinstance(model, str)

    if is_model_path:
        assert os.path.exists(model), f"Cannot access {model}. No such file or directory"
        assert os.path.isfile(model), f"{model}: Is a directory"

        model_path = os.path.abspath(model)
        model = onnx.load(model_path, load_external_data=False)

        model_dir = os.path.dirname(model_path)
        model_name = os.path.splitext(os.path.basename(model_path))[0]

    try:
        model = OnnxModelHelper.symbolic_shape_inference(model)
    except (ImportError, onnx.checker.ValidationError) as e:
        # If either onnxruntime is not available(ImportError)
        # Or if external data is not loaded for the model (onnx.checker.ValidationError)
        model = OnnxModelHelper.shape_inference(model)
        if not is_model_path and isinstance(e, onnx.checker.ValidationError):
            logger.warning("EXTERNAL DATA NOT LOADED FOR MODEL! It is the responsibility of the user to store the split .onnx file and the external data of the original model in the same directory")


    graph = gs.import_onnx(model)
    node_idx = {node.name: i for i, node in enumerate(graph.nodes)}

    def can_visit(src, dst):
        dst_idx = node_idx[dst.name]
        if node_idx[src.name] > dst_idx:
            return False

        queue = deque([src])
        while queue:
            curr = queue.popleft()
            if curr == dst:
                return True

            consumers = [
                c
                for output in curr.outputs
                for c in output.outputs
                if node_idx[c.name] <= dst_idx
            ]
            if dst in consumers:
                return True

            queue.extend(consumers)

        return False

    def get_residual_add(node):
        if node.op != "Add":
            return None
        try:
            a, b = node.i(), node.i(1)
            if a.op == "Add" and can_visit(a, b):
                return a
            elif b.op == "Add" and can_visit(b, a):
                return b
            else:
                return None
        except IndexError:
            return None

    def get_split_inputs_outputs(start_node, end_node):
        inputs, outputs = [], []
        start_idx = graph.nodes.index(start_node)
        end_idx = graph.nodes.index(end_node)

        for node in graph.nodes[start_idx : end_idx + 1]:
            for inp in node.inputs:
                if inp in graph.inputs and inp not in inputs:
                    inputs.append(inp)
            for op in node.outputs:
                if op in graph.outputs and op not in outputs:
                    outputs.append(op)

        return inputs, outputs


    residual_adds = []
    for node in graph.nodes:
        if add0 := get_residual_add(node):
            if len(residual_adds) == 0 or add0 != residual_adds[-1]:
                residual_adds.append(node)

    # lm_head split is a residual add
    n_possible_splits = len(residual_adds) + int(split_embedding) + 1    
    if n_possible_splits < num_splits:
        logger.error("Not enough layers in the model to properly split")
        return []


    embedding = []
    graph_inputs = {inp.name: inp for inp in graph.inputs}
    if split_embedding:
        start_tensor = None
        try:
            input_ids = graph_inputs["input_ids"]
            start_tensor = input_ids.o()
            embedding.append((input_ids, start_tensor))
        except KeyError:
            for node in graph.nodes:
                if node.op == "Gather" and ((start_tensor:=node.inputs[1]) in graph.inputs or (start_node:=node.inputs[2]) in graph.inputs):
                    break
            if not start_tensor:
                raise ValueError("`split_embedding` set to True, but no input named input_ids or no model input to a Gather op")
        num_splits -= 1
    else:
        try:
            start_tensor = [inp for inp in graph.inputs if inp.name in ['input_ids', 'input_embeds', 'sample']][0]
        except IndexError:
            raise ValueError("No input named 'input_ids', 'input_embeds' or 'sample' in the model")

    lm_head = []
    if split_lm_head:
        lm_head.append((residual_adds.pop().outputs[0], graph.nodes[-1].outputs[0]))
        num_splits -= 1

    # Not counting split_lm_head as it is the last residual add
    interval = len(residual_adds) / num_splits 

    split_nodes = []
    for i in range(1, num_splits):
        idx = math.floor(i * interval)
        split_nodes.append((start_tensor, residual_adds[idx].outputs[0]))
        start_tensor = residual_adds[idx].outputs[0]

    if split_lm_head:
        split_nodes.append((start_tensor, lm_head[0][0]))
    else:
        split_nodes.append((start_tensor, graph.nodes[-1].outputs[0]))


    split_nodes = embedding + split_nodes + lm_head

    splits = []

    for i, (start_tensor, end_tensor) in enumerate(split_nodes):
        start_node = start_tensor.outputs[0]
        end_node = end_tensor.inputs[0]
        inputs, outputs = get_split_inputs_outputs(start_node, end_node)

        if i != 0:
            inputs.append(start_tensor)
        if i != len(split_nodes) - 1:
            outputs.append(end_tensor)

        start_idx = node_idx[start_node.name]
        end_idx = node_idx[end_node.name]

        split = gs.Graph(
            nodes=graph.nodes[start_idx : end_idx + 1],
            inputs=inputs,
            outputs=outputs,
            name=f"split_{i + 1}",
            opset=graph.opset
        )

        splits.append(split)

    if log_level == "debug":
        table = create_rich_table(title = bold_text("Model Splitting Results:",
                              color=PrettyPrintConstants.Q_BLUE),
                              headers = ["Split Number", "New Inputs", "New Outputs"],
                              positions=[0.18, 0.59, 1.0],
                              alignment = ["left", "left", "left"])
        for split in splits:
            table.add_row(str(i+1), ', '.join([input.name for input in split.inputs]), ', '.join([output.name for output in split.outputs]))

        console = rich.console.Console(highlight=True)
        console.print(table, overflow = "fold")

    # Fill missing encodings of embedding output
    if split_embedding and encodings:
        aimet_enc = AimetEncodings(encodings)
        embedding_tensor = splits[0].outputs[0]
        start_tensor = embedding_tensor

        found, _ = aimet_enc.encoding_exists(start_tensor.name)

        while not found:
            try:
                start_tensor = start_tensor.i()
            except IndexError:
                logger.warning("Unable to fill missing encodings for Embedding layers")
                break
            found, _ = aimet_enc.encoding_exists(start_tensor.name)

        if found:
            aimet_enc.copy_encoding(start_tensor.name, embedding_tensor.name)

    splits = [gs.export_onnx(split.cleanup().toposort()) for split in splits]

    if is_model_path:
        split_paths = []

        for i, split_model in enumerate(splits):
            path = os.path.join(model_dir, f"{model_name}_split_{i+1}_of_{len(splits)}.onnx")
            onnx.save(split_model, path)
            split_paths.append(path)

    if not skip_verification:
        logger.info("Generating random inputs for model")
        random_inputs = generate_random_inputs(model)

        logger.info("Generating golden outputs for full model")
            
        output_names = [output.name for output in model.graph.output]

        if is_model_path:
            goldens = ou.run_model_on_ort(model_path, random_inputs, output_names)
        else:
            with tempfile.TemporaryDirectory() as tmpdir:
                temp_model_path = os.path.join(tmpdir, "model.onnx")
                onnx.save(model, temp_model_path, save_as_external_data=True)
                try:
                    goldens = ou.run_model_on_ort(temp_model_path, random_inputs, output_names)
                    onnx.load_external_data_for_model(model, tmpdir)
                except onnxruntime.capi.onnxruntime_pybind11_state.RuntimeException: # No external data loaded
                    logger.warning("External data not loaded for model. Skipping ONNXRT validation!")

        logger.info("Comparing outputs of split models to outputs of full model by running on ONNXRT")

        split_outputs = {}

        with tempfile.TemporaryDirectory() as tmpdir:

            for i, split_model in enumerate(splits):
                if is_model_path:
                    path = split_paths[i]
                else:
                    path = os.path.join(tmpdir, "model.onnx")
                    onnx.save(split_model, path, save_as_external_data=True)

                split_input_names = [inp.name for inp in split_model.graph.input]
                split_output_names = [output.name for output in split_model.graph.output]

                split_inputs = {
                    input_name: np_input for input_name, np_input in random_inputs.items() if input_name in split_input_names
                }

                # Output of previous split as input to this split
                for out in split_outputs:
                    if out in split_input_names:
                        split_inputs[out] = split_outputs[out]

                outputs = ou.run_model_on_ort(path, split_inputs, split_output_names)
                split_outputs.update(outputs)

                if not is_model_path: 
                    onnx.load_external_data_for_model(split_model, tmpdir)

        status = True
        for tensor_name, tensor in goldens.items():
            split_output = split_outputs[tensor_name]
            logger.debug(f"Output name: {tensor_name}. MAD: {np.abs(tensor - split_output).max()}")
            if not np.allclose(tensor, split_output, atol=1e-4):
                logger.warning(f"{Colors.FAIL}Output name: {tensor_name}. MAD: {np.abs(tensor - split_output).max()}{Colors.ENDC}")
                status = False
                break

        verification_str = (
            f"{Colors.OKGREEN if status else Colors.FAIL}"
            f"{'OK' if status else 'FAIL'}{Colors.ENDC}"
        )
        logger.info(f"Verification Status ----- {verification_str} -----")

    if is_model_path:
        logger.info(f"Input model path: {model_path}")
        logger.info(f"Split models saved at: {split_paths}")
        return split_paths
    else:
        return splits
