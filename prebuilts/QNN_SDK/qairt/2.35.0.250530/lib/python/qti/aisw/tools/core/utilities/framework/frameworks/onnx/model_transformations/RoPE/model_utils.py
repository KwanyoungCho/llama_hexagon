# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# Not a contribution.
#
# -----------------------------------------------------------------------------

from collections import deque
from onnx import helper
import numpy as np

class model_utils:
    def __init__(self, model, node_map, graph_inputs, consumer_map, value_info):
        self.graph = model.graph
        self.producer_map = {output: id(node) for node in self.graph.node for output in node.output}
        self.upstream_nodes = {id(node) : [self.producer_map[input] for input in node.input if input in self.producer_map] for node in self.graph.node}
        self.node_name_suffix_map = {} #TODO: combine with MHA2SHA
        self.initializer = {initializer.name: initializer for initializer in model.graph.initializer}
        self.consumer_map = consumer_map
        self.downstream_nodes = {
            id(node): [self.consumer_map[output] for output in node.output if output in self.consumer_map] for node in
            self.graph.node}
        self.node_map = node_map
        self.graph_inputs = graph_inputs
        self.value_info = value_info

    def get_all_upstream(self, node_id, stopping_nodes=set()):
        all_upstream = {}
        upstream_inputs = set()
        node_queue = deque([node_id])
        while node_queue:
            node = node_queue.popleft()
            if node in stopping_nodes or node in all_upstream:
                continue
            all_upstream[node] = self.node_map[node].op_type
            if node in self.upstream_nodes and self.upstream_nodes[node]:
                node_queue.extend(self.upstream_nodes[node])
            for input in self.node_map[node].input:
                if input in self.graph_inputs:
                    upstream_inputs.add(input)
        return all_upstream, upstream_inputs

    def match_pattern(self, node_id, upstream_types: list[str]):
        upstream = self.upstream_nodes[node_id].copy()
        if len(upstream_types) > len(upstream):
            return []
        found_nodes = []
        for type in upstream_types:
            found = False
            for upstream_id in upstream:
                if self.node_map[upstream_id].op_type == type:
                    found_nodes.append(upstream_id)
                    found = True
                    upstream.remove(upstream_id)
                    continue
            if not found:
                return []
        return found_nodes

    def find_first_of_above(self, node_id, op_type):
        node_queue = deque([node_id])
        while node_queue:
            node = node_queue.popleft()
            if self.node_map[node].op_type == op_type:
                return True, node
            node_queue.extend(self.upstream_nodes[node])
        return False, -1

    def find_first_of_below(self, node_id, op_type):
        node_queue = deque([node_id])
        while node_queue:
            node = node_queue.popleft()
            if self.node_map[node].op_type == op_type:
                return True, node
            node_queue.extend(*self.downstream_nodes[node])
        return False, -1

    def create_node_name(self, op_type,name_prefix = None,):
        if name_prefix:
            prefix = name_prefix if name_prefix.endswith("_") else (name_prefix + "_")
        else:
            prefix = op_type + "_"
        suffix: int = 0
        if prefix in self.node_name_suffix_map:
            suffix = self.node_name_suffix_map[prefix] + 1
        else:
            for node in self.graph.node:
                if node.name and node.name.startswith(prefix):
                    try:
                        index = int(node.name[len(prefix):])
                        suffix = max(index + 1, suffix)
                    except ValueError:
                        continue
        self.node_name_suffix_map[prefix] = suffix
        return prefix + str(suffix)

    def get_seq_len(self):
        if 'input_ids' in self.graph_inputs:
            input_ids, = [i for i in self.graph.input if i.name == 'input_ids']
            _, seq_len = [i.dim_value for i in input_ids.type.tensor_type.shape.dim]
        elif 'input_embeds' in self.graph_inputs:
            input_embeds, = [i for i in self.graph.input if i.name == 'input_embeds']
            _, seq_len, _ = [i.dim_value for i in input_embeds.type.tensor_type.shape.dim]
        return seq_len

    def get_reshape_constant(self, reshape):
        reshape_input = self.node_map[reshape].input[1]
        if reshape_input in self.initializer:
            initializer = self.initializer[reshape_input]
            np_dtype = helper.tensor_dtype_to_np_dtype(initializer.data_type)
            return np.frombuffer(initializer.raw_data, dtype=np_dtype).reshape(initializer.dims)
        elif self.node_map[self.producer_map[reshape_input]].op_type == "Constant":
            tensor = self.node_map[self.producer_map[reshape_input]].attribute[0].t
            np_dtype = helper.tensor_dtype_to_np_dtype(tensor.data_type)
            return np.frombuffer(tensor.raw_data, dtype=np_dtype).reshape(tensor.dims)
        return None