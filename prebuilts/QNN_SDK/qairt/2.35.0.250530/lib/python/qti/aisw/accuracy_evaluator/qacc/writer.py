# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import cv2

from qti.aisw.accuracy_evaluator.qacc.constants import Constants as qcc
import qti.aisw.accuracy_evaluator.common.exceptions as ce


class Writer:

    def __init__(self):
        pass

    def write(self, output_path, mem_obj, dtype, write_format='npy'):
        """Use a specific writer to write the mem_obj The write methods needs
        to be thread safe."""

        if write_format == qcc.FMT_CV2:
            return CV2Writer.write(output_path, mem_obj)
        elif write_format == qcc.FMT_PIL:
            return PILWriter.write(output_path, mem_obj)
        elif write_format == qcc.FMT_NPY:
            return RawWriter.write(output_path, mem_obj, dtype)

        raise ce.UnsupportedException('Invalid Writer type : ' + write_format)


class CV2Writer:

    @classmethod
    def write(self, output_path, mem_obj):
        cv2.imwrite(output_path, mem_obj)


class PILWriter:

    @classmethod
    def write(self, output_path, mem_obj):
        mem_obj.save(output_path)


class RawWriter:

    @classmethod
    def write(self, output_path, mem_obj, dtype):
        mem_obj.astype(dtype).tofile(output_path)
