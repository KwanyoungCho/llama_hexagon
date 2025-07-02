# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

from typing import List, Union

import numpy as np
from torch.utils.data import DataLoader, Dataset


def _convert_to_list(
    data: Union[Dataset, DataLoader], batch_size: int, num_of_samples: int
) -> List[np.ndarray]:
    """
    Iterates over a Dataset or a DataLoader and converts each sample into a numpy array.

    Args:
        data: A Pytorch Dataset or DataLoader
        batch_size: The batch size to use if 'data' is a Dataset.
        num_of_samples: If provided, only the first 'num_of_samples' batches/samples are
            processed.

    Returns:
        A list where each element is a numpy array
    """

    if isinstance(data, Dataset):
        dataloader = DataLoader(data, batch_size=batch_size)
    else:
        dataloader = data

    inputs_list = []
    total_samples = 0

    for sample in dataloader:
        if total_samples >= num_of_samples:
            break

        # Unpack the batch if it's a list of tuples
        if isinstance(sample[0], list) or isinstance(sample[0], tuple):
            sample = sample[0]

        # Determine the dtype from the first item in the sample
        dtype = sample[0].numpy().dtype

        # Convert each item in the batch to a NumPy array with the desired data type
        inputs = [np.array(item.numpy(), dtype=dtype) for item in sample]
        inputs_list.extend(inputs)
        total_samples += len(sample)

    return inputs_list
