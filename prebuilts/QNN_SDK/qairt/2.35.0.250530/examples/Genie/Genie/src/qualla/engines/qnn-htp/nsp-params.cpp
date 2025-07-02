//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "fmt/format.h"
#include "nsp-params.hpp"
#include "qualla/detail/config.hpp"

namespace qualla {
// Helper functions to convert enum to string
NLOHMANN_JSON_SERIALIZE_ENUM(RopeScalingParams::RopeType,
                             {{RopeScalingParams::DEFAULT, "default"},
                              {RopeScalingParams::ROPE_LLAMA3, "llama3"},
                              {RopeScalingParams::ROPE_LONGROPE, "longrope"}})

NLOHMANN_JSON_SERIALIZE_ENUM(PositionalEncoding::EncodingType,
                             {{PositionalEncoding::UNDEFINED, "undefined"},
                              {PositionalEncoding::ROPE, "rope"},
                              {PositionalEncoding::ABSOLUTE, "absolute"},
                              {PositionalEncoding::ALIBI, "alibi"}})

NLOHMANN_JSON_SERIALIZE_ENUM(LongContextParams::Mode,
                             {{LongContextParams::DISABLED, "disabled"},
                              {LongContextParams::SLIDING_WINDOW, "sliding-window"},
                              {LongContextParams::KEYDIFF, "keydiff"}})

// Utility functions to convert structs from/to json for parsing/dumping
void from_json(const json& j, RopeScalingParams& p) {
  p.rope_type = Config::optional(j, "rope-type", RopeScalingParams::DEFAULT);
  if (p.rope_type == RopeScalingParams::ROPE_LLAMA3) {
    try {
      j.at("factor").get_to(p.llama3_params.factor);
      j.at("low-freq-factor").get_to(p.llama3_params.low_freq_factor);
      j.at("high-freq-factor").get_to(p.llama3_params.high_freq_factor);
      j.at("original-max-position-embeddings")
          .get_to(p.llama3_params.original_max_position_embeddings);
    } catch (const json::exception& e) {
      throw std::runtime_error(
          fmt::format("Parsing error for llama3 rope scaling - {}\n"
                      "llama3 requires keys ['original-max-position-embeddings', 'factor', "
                      "'low-freq-factor', 'high-freq-factor'].\n"
                      "Found config - {}",
                      e.what(),
                      j.dump()));
    }
  } else if (p.rope_type == RopeScalingParams::ROPE_LONGROPE) {
    try {
      j.at("original-max-position-embeddings")
          .get_to(p.longrope_params.original_max_position_embeddings);
      j.at("long-factor").get_to(p.longrope_params.long_factor);
      j.at("short-factor").get_to(p.longrope_params.short_factor);
      if (j.contains("factor"))
        j.at("factor").get_to(p.longrope_params.factor);
      else
        p.longrope_params.factor = j.at("max-position-embeddings").get<double>() /
                                   p.longrope_params.original_max_position_embeddings;
    } catch (const json::exception& e) {
      throw std::runtime_error(
          fmt::format("Parsing error for longrope scaling - {}\n"
                      "LongRope requires keys ['original-max-position-embeddings', 'factor' or "
                      "'max-position-embeddings', 'long-factor', 'short-factor'].\n"
                      "Found config - {}",
                      e.what(),
                      j.dump()));
    }
  }
}

void to_json(json& j, const RopeScalingParams& p) {
  j["rope-type"] = p.rope_type;
  if (p.rope_type == RopeScalingParams::ROPE_LLAMA3) {
    j["factor"]                           = p.llama3_params.factor;
    j["low-freq-factor"]                  = p.llama3_params.low_freq_factor;
    j["high-freq-factor"]                 = p.llama3_params.high_freq_factor;
    j["original-max-position-embeddings"] = p.llama3_params.original_max_position_embeddings;
  } else if (p.rope_type == RopeScalingParams::ROPE_LONGROPE) {
    j["factor"]                           = p.longrope_params.factor;
    j["long-factor"]                      = p.longrope_params.long_factor;
    j["short-factor"]                     = p.longrope_params.short_factor;
    j["original-max-position-embeddings"] = p.longrope_params.original_max_position_embeddings;
  }
}

void from_json(const json& j, PositionalEncoding& p) {
  p.type = Config::optional(j, "type", PositionalEncoding::ROPE);
  if (p.type == PositionalEncoding::ROPE) {
    p.rope_params.dims         = Config::mandatory<int32_t>(j, "rope-dim");
    p.rope_params.theta        = Config::optional<int32_t>(j, "rope-theta", 10000);
    p.rope_params.rope_scaling = Config::optional<RopeScalingParams>(j, "rope-scaling", {});
  }
}

void to_json(json& j, const PositionalEncoding& p) {
  j["type"] = p.type;
  if (p.type == PositionalEncoding::ROPE) {
    j["rope-dim"]     = p.rope_params.dims;
    j["rope-theta"]   = p.rope_params.theta;
    j["rope-scaling"] = p.rope_params.rope_scaling;
  }
}

void from_json(const json& j, LongContextParams& p) {
  p.mode = Config::optional(j, "mode", LongContextParams::DISABLED);

  if (p.mode != LongContextParams::DISABLED)
    p.sink_tokens = Config::optional<int32_t>(j, "sink-tokens", 0);

  if (p.mode == LongContextParams::KEYDIFF) {
    p.update_frequency = Config::optional<int32_t>(j, "update-frequency", 128);
    p.scoring_network  = Config::mandatory<std::string>(j, "scoring-network");
  }
}

void to_json(json& j, const LongContextParams& p) {
  j["mode"] = p.mode;
  if (p.mode != LongContextParams::DISABLED) j["sink-tokens"] = p.sink_tokens;
  if (p.mode == LongContextParams::KEYDIFF) {
    j["update-frequency"] = p.update_frequency;
    j["scoring-network"]  = p.scoring_network;
  }
}

}  // namespace qualla