# =============================================================================
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All rights reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# =============================================================================

import json
import logging
import os
from typing import Optional

import pandas as pd
from qti.aisw.accuracy_debugger.encodings_converter.qairt_encodings_converter import (
    QairtEncodingsConverter,
)
from qti.aisw.accuracy_debugger.utils.exceptions import (
    EncodingsMismatchError,
    QairtEncodingsConverterFailure,
)
from qti.aisw.tools.core.utilities.qairt_logging.log_areas import LogAreas
from qti.aisw.tools.core.utilities.qairt_logging.logging_utility import QAIRTLogger


class CompareEncodings:
    """The CompareEncodings class provides essential functions to compare two encoding files,
    one being a reference. It filters out QAIRT encodings by removing conversion operations
    inserted by QAIRT for datatype conversion, which aren't present in the reference encoding.
    Mismatched encodings are documented in an Excel sheet, and the path to this file is returned.
    """

    def __init__(self, logger: logging.Logger = None) -> None:
        """Initializes the Compare Encodings class

        Args:
            logger: Python Logger object
        """
        if logger is None:
            log_area = LogAreas.register_log_area("Compare Encodings")
            self._logger = QAIRTLogger.register_area_logger(
                area=log_area, level="INFO", formatter_val="simple", handler_list=["dev_console"]
            )
        else:
            self._logger = logger
        self.output_dir = None

    def run(
        self,
        qairt_encodings_path: str,
        reference_encodings_path: str,
        output_dir: str,
        quantized_dlc_path: str,
        framework_model: str,
        ignore_encodings_error: bool = True,
    ) -> str:
        """Execute Compare Encodings

        Args:
            qairt_encodings_path: Path to QAIRT encodings dumped from quantizer
            reference_encodings_path: Path to external reference encodings
            output_dir: Directory path to store the results
            quantized_dlc_path: Path to the quantized DLC file associated with the given encodings
            framework_model: Path to the framework model
            ignore_encodings_error: Whether to ignore exceptions due to encodings mismatch and
                                    continue execution. Defaults to True.

        Return:
            str: Path to the Excel file with the differences between reference and QAIRT encodings

        Raises:
            FileNotFoundError: If the reference encodings file is not found.
            Exception: If there is an error decoding the JSON file or writing the filtered
                       encodings to a file.
        """
        self.output_dir = output_dir

        # Load reference encodings
        try:
            with open(reference_encodings_path) as json_file:
                reference_encodings = json.load(json_file)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Reference encodings file not found at {reference_encodings_path}"
            )
        except Exception as exception:
            raise Exception(f"Error loading reference encodings: {exception}")

        # Filter encodings based on given paths and models
        filtered_encodings = self._filter_encodings(
            qairt_encodings_path, quantized_dlc_path, framework_model
        )

        filtered_encodings_file_path = os.path.join(self.output_dir, "filtered_encodings.json")
        try:
            with open(filtered_encodings_file_path, "w") as fh:
                json.dump(filtered_encodings, fh, indent=4)
        except Exception as exception:
            raise Exception(f"Error writing filtered encodings to file: {exception}")

        # Generate excel sheet highlighting any mismatches between reference and QAIRT encodings
        encoding_diff_df = self._generate_dataframe(
            reference_encodings, filtered_encodings, ignore_encodings_error
        )
        encoding_diff_path = os.path.join(output_dir, "encodings_diff.xlsx")
        self._save_dataframe_to_excel(encoding_diff_df, encoding_diff_path)

        # Log warnings if any encodings are present in reference but not in QAIRT and vice-versa
        self._check_missing_encodings(
            qairt_encodings=filtered_encodings, reference_encodings=reference_encodings
        )
        self._logger.info(
            f"Differences in QAIRT encodings and reference encodings are written to {encoding_diff_path}"
        )
        return encoding_diff_path

    def _check_missing_encodings(self, qairt_encodings: dict, reference_encodings: dict) -> None:
        """Identify encodings present in reference but not in target (QAIRT) and vice-versa.

        Args:
            qairt_encodings: QAIRT encodings dumped by quantizer
            reference_encodings: Reference/external encodings
        """
        self._logger.info(
            "Finding encodings present only in reference encodings but not in QAIRT encodings"
        )
        for enc_type in reference_encodings:
            if enc_type in qairt_encodings:
                self._logger.debug(f"Checking {enc_type}...")
                for layer in reference_encodings[enc_type]:
                    if all(
                        alias not in qairt_encodings[enc_type]
                        for alias in [layer, layer + "_permute"]
                    ):
                        self._logger.warning(f"{layer} present only in reference encodings")
            else:
                self._logger.warning(f"{enc_type} present only in reference encodings")

        self._logger.info(
            "Finding encodings present only in QAIRT encodings but not in reference encodings"
        )
        for enc_type in qairt_encodings:
            if enc_type in reference_encodings:
                self._logger.debug(f"Checking {enc_type}...")
                for layer in qairt_encodings[enc_type]:
                    if all(
                        alias not in reference_encodings[enc_type]
                        for alias in [layer, layer.replace("_permute", "")]
                    ):
                        self._logger.warning(f"{layer} present only in QAIRT encodings")
            else:
                self._logger.warning(f"{enc_type} present only in QAIRT encodings")

    def _filter_encodings(
        self, qairt_encodings_path: str, quantized_dlc_path: str, framework_model: str
    ) -> dict:
        """Filter the QAIRT encodings by removing convert ops inserted by QAIRT for datatype
        conversion. They are removed as they are not present in reference encodings, therefore
        they should not be compared directly.

        Args:
            qairt_encodings_path: Path to the QAIRT encodings JSON file
            quantized_dlc_path: Path to the quantized DLC file, which the given encodings belong to
            framework_model: Path to the framework model

        Returns:
            dict: Filtered QAIRT encodings
        """
        # create object of QairtEncodingsConverter and filter convert ops
        qairt_encodings_converter = QairtEncodingsConverter(
            framework_model, quantized_dlc_path, qairt_encodings_path, self.output_dir, self._logger
        )
        try:
            filtered_encodings = qairt_encodings_converter.create_subgraph_encodings()
        except Exception as exception:
            raise QairtEncodingsConverterFailure(
                f"Failed to filter QAIRT encodings: {exception}"
            ) from exception
        return filtered_encodings

    def _compare_specific_encoding(
        self,
        reference_encodings: dict,
        qairt_encodings: dict,
        key: str,
        correction: Optional[str],
        precision: int = 15,
    ) -> bool:
        """Compares the specific encoding and returns the comparison result and correction if needed.

        Args:
            reference_encodings: The reference encoding dictionary.
            qairt_encodings: The target encoding dictionary.
            key: The encoding to compare in the dictionaries.
            correction: The correction type if needed. It denotes whether the scale and
                offset have to be adjusted based on the bitwidth before comparing.
                Possible values for correction are:
                    - "up": Scale should be multiplied and offset divided by 256
                        (target bitwidth 16 and reference bitwidth 8).
                    - "down": Scale should be divided and offset multiplied by 256
                        (target bitwidth 8 and reference bitwidth 16).
            precision: The precision for rounding. Default is 15.

        Returns:
            bool: True if encodings match, False otherwise.
        """
        if key in ["dtype", "is_symmetric"]:
            qairt_encodings[key] = str(qairt_encodings[key]).lower()
            reference_encodings[key] = str(reference_encodings[key]).lower()

        if key in ["scale", "offset"] and correction:
            if (correction == "up" and key == "offset") or (
                correction == "down" and key == "scale"
            ):
                compare_encoding = round(qairt_encodings[key], precision) == round(
                    (reference_encodings[key] * 256.0), precision
                ) or round(qairt_encodings[key], precision) == round(
                    (reference_encodings[key] * 257.0), precision
                )
            else:
                compare_encoding = round(qairt_encodings[key], precision) == round(
                    (reference_encodings[key] / 256.0), precision
                ) or round(qairt_encodings[key], precision) == round(
                    (reference_encodings[key] / 257.0), precision
                )
        elif (
            key in ["offset", "is_symmetric"]
            and qairt_encodings["is_symmetric"] == "false"
            and qairt_encodings["offset"] == 0
            and reference_encodings["is_symmetric"] == "true"
            and reference_encodings["offset"] == -128
        ):
            compare_encoding = True
        elif key in ["max", "min", "scale", "offset"]:
            compare_encoding = round(qairt_encodings[key], precision) == round(
                reference_encodings[key], precision
            )
        else:
            compare_encoding = qairt_encodings[key] == reference_encodings[key]

        return compare_encoding

    def _generate_diff_warning(
        self,
        key: str,
        reference_encoding_dict: dict,
        qairt_encoding_dict: dict,
        correction: Optional[str],
    ) -> tuple[str, Optional[str]]:
        """Generates the appropriate diff warning message based on the comparison result.

        Args:
            key: The key being compared.
            reference_encoding_dict: The reference encoding dictionary.
            qairt_encoding_dict: The QAIRT encodings dictionary.
            correction: The correction type if needed. It denotes whether the scale and
                offset have to be adjusted based on the bitwidth before comparing.
                Possible values for correction are:
                    - "up": target bitwidth 16 and reference bitwidth 8
                    - "down": target bitwidth 8 and reference bitwidth 16

        Returns:
            tuple[str, Optional[str]]: The diff warning message and updated correction type.
        """
        reference_encoding = reference_encoding_dict[key]
        qairt_encoding = qairt_encoding_dict[key]

        diff_warning = (
            f"* QAIRT encoding = {qairt_encoding} reference encoding = {reference_encoding}"
        )

        if key == "bitwidth":
            diff_warning = (
                f"| QAIRT encoding = {qairt_encoding} reference encoding = {reference_encoding}"
            )
            if qairt_encoding == 16 and reference_encoding == 8:
                correction = "up"
            elif qairt_encoding == 8 and reference_encoding == 16:
                correction = "down"
            else:
                diff_warning = f"* Activation bitwidth conversions from reference encoding = {reference_encoding} to QAIRT encoding = {qairt_encoding} not supported"
        elif key in ["scale", "offset"] and correction:
            diff_warning = f"* {key} not consistent according to bitwidth conversion QAIRT encoding = {qairt_encoding} reference encoding = {reference_encoding}"

        return diff_warning, correction

    def _add_encoding_to_dataframe(
        self,
        df: pd.DataFrame,
        encoding_type: str,
        encoding_name: str,
        reference_encoding_dict: dict,
        qairt_encoding_dict: dict,
        headers: list[str],
        ignore_encodings_error: bool = True,
    ) -> pd.DataFrame:
        """Adds encoding data to the DataFrame.

        Args:
            df: The DataFrame to which the data will be added.
            encoding_type: The type of encoding.
            encoding_name: The name of the encoding.
            reference_encoding_dict: The reference encoding dictionary.
            qairt_encoding_dict: The QAIRT encodings dictionary.
            headers: A list of headers for the Excel file.
            ignore_encodings_error: If True, ignores exceptions raised due to encoding mismatches
                between QAIRT encodings and reference encodings and continues execution.

        Returns:
            pd.DataFrame: The updated DataFrame with the new encoding data added.
        """
        row_data = {header: None for header in headers}
        row_data["Encoding_type"] = encoding_type
        row_data["buffer_name"] = encoding_name
        correction = None
        encoding_mismatches = 0

        for key in reference_encoding_dict.keys():
            if key not in qairt_encoding_dict.keys():
                continue

            compare_encoding = self._compare_specific_encoding(
                reference_encoding_dict, qairt_encoding_dict, key, correction
            )
            ref_encoding = reference_encoding_dict[key]
            qairt_encoding = qairt_encoding_dict[key]
            if not compare_encoding:
                if not ignore_encodings_error:
                    raise EncodingsMismatchError(
                        f"Encodings mismatch observed in {encoding_type} for {encoding_name} in {key}."
                        f"QAIRT Encoding: {qairt_encoding} Reference Encoding: {ref_encoding}"
                    )
                encoding_mismatches += 1
                diff_warning, correction = self._generate_diff_warning(
                    key, reference_encoding_dict, qairt_encoding_dict, correction
                )
                row_data[key] = diff_warning
            else:
                row_data[key] = str(qairt_encoding)

        if encoding_mismatches:
            df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)

        return df

    def _save_dataframe_to_excel(self, df: pd.DataFrame, encoding_diff_path: str) -> None:
        """Saves the provided DataFrame to an Excel file at the specified path.

        Args:
            df: The DataFrame to save.
            encoding_diff_path: The path where the Excel file will be saved.
        """

        def highlight_cells(val):
            color = ""
            if str(val).startswith("|"):
                color = "blue"
            elif str(val).startswith("*"):
                color = "red"
            return f"color: {color}"

        writer = pd.ExcelWriter(encoding_diff_path, engine="xlsxwriter")
        workbook = writer.book
        worksheet = workbook.add_worksheet("Sheet1")
        bold_format = workbook.add_format({"bold": True})

        # Add a description on the top of the file explaining the color scheme for the text
        worksheet.write(0, 0, "Text Color Scheme", bold_format)
        color_description = [
            ("Black", "Contains the matched encoding in case of no mismatch"),
            ("Red", "Represents errors in case of encodings mismatch"),
            ("Blue", "Represents warnings in case of bitwidth mismatch"),
        ]

        for i, (color, description) in enumerate(color_description, start=1):
            worksheet.write(
                i,
                0,
                color,
            )
            worksheet.write(i, 1, description)

        # The encodings data will be dumped below the description
        start_row = len(color_description) + 1
        df.style.applymap(highlight_cells).to_excel(
            writer, sheet_name="Sheet1", startrow=start_row, index=False
        )
        writer.close()

    def _generate_dataframe(
        self,
        reference_encodings: dict,
        qairt_encodings: dict,
        ignore_encodings_error: bool = True,
    ) -> pd.DataFrame:
        """Generate an dataframe from comparing reference and target encodings.

        Args:
            reference_encodings: Reference encodings dictionary.
            qairt_encodings: QAIRT encodings dictionary.
            ignore_encodings_error: Ignore the exception raised due to encodings mismatch
                between QAIRT encodings and reference encodings and continue execution

        Returns:
            pd.DataFrame : Dataframe containing encoding matches and mismatches
        """
        headers = ["Encoding_type", "buffer_name", "bitwidth", "dtype", "is_symmetric", "max",
                   "min", "offset", "scale"]  # fmt: skip

        df = pd.DataFrame(columns=headers)
        diff_counts = {}

        for encoding_type in reference_encodings.keys():
            diff_counts[encoding_type] = 0

            if encoding_type not in qairt_encodings.keys():
                continue

            for encoding_name, reference_encoding_list in reference_encodings[
                encoding_type
            ].items():
                if encoding_name not in qairt_encodings[encoding_type].keys():
                    continue
                for idx, reference_encoding_dict in enumerate(reference_encoding_list):
                    qairt_encoding_dict = qairt_encodings[encoding_type][encoding_name][idx]
                    df = self._add_encoding_to_dataframe(
                        df,
                        encoding_type,
                        encoding_name,
                        reference_encoding_dict,
                        qairt_encoding_dict,
                        headers,
                        ignore_encodings_error,
                    )

        return df
