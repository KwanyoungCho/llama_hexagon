# -----------------------------------------------------------------------------
#
# Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
# All Rights Reserved.
# Confidential and Proprietary - Qualcomm Technologies, Inc.
#
# Not a contribution.
#
# -----------------------------------------------------------------------------

import json
from enum import Enum

from .logger import log_verbose, setup_logging


class AimetEncodingVersion(str, Enum):
    V0_6_1 = "0.6.1"
    V1_0_0 = "1.0.0"


class AimetEncodings:
    def __init__(self, encodings: str | dict, log_level: str = "info"):
        setup_logging(log_level)

        if isinstance(encodings, str):
            log_verbose(f"Loading encodings file: {encodings}")
            with open(encodings) as json_file:
                quant_encoding_dict = json.load(json_file)
            self.encodings = quant_encoding_dict
        else:
            self.encodings = encodings

        if "version" not in self.encodings:
            raise ValueError(f"Encodings file does not have field 'version'")

        self.version = self.encodings["version"]

        if self.version not in AimetEncodingVersion._value2member_map_:
            raise ValueError(f"Encodings file has unsupported version")

        self.activation_enc = self.encodings["activation_encodings"]
        self.param_enc = self.encodings["param_encodings"]

        self.changed = False

    def encoding_exists(self, encoding):
        if self.version == AimetEncodingVersion.V0_6_1:
            if encoding in self.activation_enc or encoding in self.param_enc:
                return True, self.activation_enc[
                    encoding
                ] if encoding in self.activation_enc else self.param_enc[encoding]
            else:
                return False, None
        else:
            found_enc = list(filter(lambda x: x["name"] == encoding, self.param_enc + self.activation_enc))
            if len(found_enc) != 0:
                return True, found_enc[0]
            else:
                return False, None

    def copy_encoding(self, original_name: str, new_name: str, param_enc: bool = False):
        _, original_enc = self.encoding_exists(original_name)
        if not original_enc:
            raise ValueError(f"Could not get encodings for {original_name}")

        log_verbose(f"Copy encoding {original_name} -> {new_name}")
        if self.version == AimetEncodingVersion.V0_6_1:
            if param_enc:
                self.param_enc[new_name] = original_enc
            else:
                self.activation_enc[new_name] = original_enc

        elif self.version == AimetEncodingVersion.V1_0_0:
            new_enc = original_enc.copy()
            new_enc["name"] = new_name

            if param_enc:
                enc_idx = [idx for idx, enc in enumerate(self.param_enc) if enc["name"] == new_name]
                if len(enc_idx) != 0:
                    self.param_enc.pop(enc_idx[0])
                self.param_enc.append(new_enc)
            else:
                enc_idx = [idx for idx, enc in enumerate(self.activation_enc) if enc["name"] == new_name]
                if len(enc_idx) != 0:
                    self.activation_enc.pop(enc_idx[0])
                self.activation_enc.append(new_enc)

        self.changed = True

    def save_encodings(self, file_path: str):
        with open(file_path, "wt") as json_file:
            json.dump(self.encodings, json_file, indent=4, sort_keys=True)
