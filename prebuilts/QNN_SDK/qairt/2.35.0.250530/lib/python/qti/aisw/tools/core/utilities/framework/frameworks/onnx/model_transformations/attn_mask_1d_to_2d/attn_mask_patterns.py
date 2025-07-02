# ==============================================================================
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# Not a contribution.
# ==============================================================================

attention_patterns = [
    # GPT Pattern
    {
        "pattern": ["MatMul", "Div", "Where", "Add", "Softmax"],
        "remove_nodes": ["Where"],      # We can use [-3] or [2] here.
    },
    # Baichuan Pattern
    {
        "pattern": ["MatMul", "Div", "Add", "Max", "Softmax"],
        "remove_nodes": ["Max"],        # We can use [-2] or [3] here.
    },
    # LLaMa Pattern
    {
        "pattern": ["MatMul", "Div", "Add", "Softmax"],
        "remove_nodes": [],
    },
    # LLaMa Pattern
    {
        "pattern": ["Div", "MatMul", "Add", "Softmax"],
        "remove_nodes": [],
    },
]
