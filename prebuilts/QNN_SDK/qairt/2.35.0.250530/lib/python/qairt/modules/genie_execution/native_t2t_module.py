# ==============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# ==============================================================================

import json
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from pathlib import Path
from typing import Callable, List, Union

import numpy as np

from qairt.gen_ai_api.executors.gen_ai_executor import GenerationExecutionResult, GenerationMetrics
from qairt.modules.genie_execution import genie
from qairt.modules.genie_execution.genie_config import GenieConfig
from qairt.modules.genie_execution.genie_profile_record import (
    ComponentType,
    DialogCreateEvent,
    DialogQueryEvent,
    GenieProfileRecord,
)
from qairt.utils import loggers


class GenieNativeT2TRunner:
    _logger = loggers.get_logger(name=__name__)

    def __init__(self, genie_config: GenieConfig, query_timeout: int = 360):
        """
        The GenieNativeT2TRunner can be constructed from the same GenieConfig used with the command line tool and optionally
        the user may set the query timeout (in seconds).

        Args:
            genie_config (GenieConfig): GenieConfig defining model and execution configuration
            query_timeout (int): Timeout period after which the abort signal will be sent terminating a query
        """
        self._profile_config: genie.ProfileConfig = genie.ProfileConfig()
        self._profile: genie.Profile = genie.Profile(self._profile_config)
        self._dialog_config: genie.DialogConfig = genie.DialogConfig(str(genie_config))
        self._dialog_config.bind_profile(self._profile)
        self._dialog: genie.Dialog = genie.Dialog(self._dialog_config)
        self._query_timeout: int = query_timeout
        self._sampler: genie.Sampler = self._dialog.get_sampler()
        self._sampler_callbacks: List[Callable[[np.ndarray], int]] = []
        self._sampler_configs: List[str] = []

    def __del__(self):
        del self._dialog
        del self._dialog_config
        del self._profile
        del self._profile_config
        del self._sampler

    def query(self, prompt: str) -> GenerationExecutionResult:
        """
        Executes the provided query

        Args:
            prompt (str): The string prompt to be
        Returns:
            GenerationExecutionResult: Generated text and execution metrics from native execution
        """
        out = GenerationExecutionResult()

        output = []
        started = False

        def capture_output(response: str, code: genie.GenieDialogSentenceCode):
            nonlocal output
            nonlocal started
            started = True
            output.append((response, code))

        with ThreadPoolExecutor(max_workers=1) as executor:
            self._logger.debug(f"querying dialog with: {prompt}")
            future = executor.submit(self._dialog.query, prompt, capture_output)
            try:
                future.result(timeout=self._query_timeout)
            except genie.GenieException as e:
                out.error = str(e)
                return out
            except TimeoutError:
                out.error = f"Query timed out for prompt: {prompt}"
                self._logger.warning(f"Sending abort signal as query timed out for prompt: {prompt}")
                while not started:
                    pass
                self._dialog.signal(genie.GenieDialogAction.ABORT)
                return out

        for response, code in output:
            if code in {genie.GenieDialogSentenceCode.BEGIN, genie.GenieDialogSentenceCode.CONTINUE}:
                out.generated_text += response
            if code == genie.GenieDialogSentenceCode.ABORT:
                out.error += genie.GenieDialogSentenceCode.ABORT.name
        profile_record = GenieProfileRecord(**json.loads(self._profile.get_json_data()))

        dialog_component = [x for x in profile_record.components if x.type == ComponentType.DIALOG][0]
        dialog_create_event = [x for x in dialog_component.events if isinstance(x, DialogCreateEvent)][0]
        dialog_query_event = [x for x in dialog_component.events if isinstance(x, DialogQueryEvent)][-1]
        out.metrics = GenerationMetrics(
            init_time=int(dialog_create_event.init_time.value) if dialog_create_event.init_time else None,
            prompt_processing_time=int(
                dialog_query_event.num_prompt_tokens.value * dialog_query_event.prompt_processing_rate.value
            )
            if dialog_query_event.num_prompt_tokens and dialog_query_event.prompt_processing_rate
            else None,
            prompt_processing_rate=dialog_query_event.prompt_processing_rate.value
            if dialog_query_event.prompt_processing_rate
            else None,
            token_generation_time=int(dialog_query_event.token_generation_time.value)
            if dialog_query_event.token_generation_time
            else None,
            token_generation_rate=dialog_query_event.token_generation_rate.value
            if dialog_query_event.token_generation_rate
            else None,
        )
        return out

    def save_dialog(self, save_dir: Union[str, os.PathLike]) -> None:
        """
        Stores the current state of the genie dialog

        Args:
            save_dir (Union[str, os.PathLike]): Location to save the dialog to
        """
        path = Path(save_dir)
        if path.is_file():
            raise FileExistsError(
                f"Provided save location is an existing file {path}. Please provide a directory"
            )
        path.mkdir(exist_ok=True, parents=True)

        self._dialog.save(str(path))
        self._logger.info(f"Saved dialog to {path}")

    def restore_dialog(self, saved_dialog: Union[str, os.PathLike]) -> "GenieNativeT2TRunner":
        """
        Restores a saved genie dialog state

        Args:
            saved_dialog (Union[str, os.PathLike]): Path to saved dialog state to restore
        Returns:
            self: Returns self after restoration
        """
        path = Path(saved_dialog)
        if not path.is_dir():
            raise NotADirectoryError(f"Provided path to saved dialog {path} is not an existing directory.")

        self.reset_dialog()
        self._dialog.restore(str(path))
        self._logger.info(f"Restored dialog from {path}")
        return self

    def reset_dialog(self) -> "GenieNativeT2TRunner":
        """
        Resets dialog state to remove context accumulated from queries

        Returns:
            self: Returns self after resetting dialog state
        """
        self._dialog.reset()
        self._logger.info(f"Reset dialog")
        return self

    def register_sampler_callback(
        self, name: str, callback: Callable[[np.ndarray], int]
    ) -> "GenieNativeT2TRunner":
        """
        Register sampler callback function

        Args:
            name (str): Name of the callback. Passed in a sampler config to set the desired sampler callback
            callback (Callable[[np.ndarray], int]): Sampler callback to select the next token given logits
        Returns:
            self: Returns self after registering callback
        """
        self._sampler.register_callback(name, callback)
        self._sampler_callbacks.append(callback)
        self._logger.info("Registered sampler callback")
        return self

    def apply_sampler_config(self, config: str) -> "GenieNativeT2TRunner":
        """
        Apply a sampler config either setting sampler parameters for the default sampler or supplying the
        name of a registered sampler callback function

        Args:
            config (str): json string representing the sampler config to apply
        Returns:
            self: Returns self after applying the sampler config
        """
        sampler_config = genie.SamplerConfig(config)
        self._sampler.apply_config(sampler_config)
        self._sampler_configs.append(sampler_config)
        self._logger.info("Applied sampler config")
        return self
