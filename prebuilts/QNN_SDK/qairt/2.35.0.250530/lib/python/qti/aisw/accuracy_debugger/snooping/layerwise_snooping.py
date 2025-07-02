# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================
import logging
from pathlib import Path

from qti.aisw.accuracy_debugger.snooping.subgraph_snooper import SubgraphSnooper
from qti.aisw.accuracy_debugger.utils.constants import Algorithm


class LayerwiseSnooper(SubgraphSnooper):
    """Subclass for layerwise algorithm."""

    def __init__(self, model: Path, logger: logging.Logger):
        """Initializes the LayerwiseSnooper.

        Args:
            model (Path): Path to framework model.
            logger (logging.Logger): A python logger instance
        """
        self._activation_status = {}
        super().__init__(model=model, name=Algorithm.LAYERWISE, logger=logger, is_cumulative=False)
