//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_ENV_HPP
#define QUALLA_ENV_HPP

#include <filesystem>
#include <memory>
#include <unordered_set>

#include "GenieLog.h"
#include "LogUtils.hpp"
#include "Logger.hpp"
#include "qualla/detail/config.hpp"
#include "qualla/detail/exports.h"
#include "qualla/detail/json.hpp"
#include "qualla/detail/state.hpp"

namespace qualla {

enum class LayerType {
  INPUT,
  OUTPUT,
  ATTN_MASK,
  ANCHOR,
  VALID_MASK,
  POS_SIN,
  POS_COS,
  POS_IDS,
  CACHE_INDEX,
  TOKEN_TYPE_IDS,
  POOL_OUTPUT,
  SEQ_OUTPUT,
  INPUT_EMBED
};

enum InputType { TOKENS = 0x01, EMBEDDINGS = 0x02, PIXELS = 0x03, UNKNOWN = 0xFF };

class Env : public State {
 public:
  QUALLA_API Env(const json& conf);
  QUALLA_API ~Env();

  struct Path {
    std::filesystem::path models;
    std::filesystem::path cache;
  };

  const Path& path() const { return _path; }

  std::shared_ptr<genie::log::Logger> logger() {
    if (_logger.size() > 0) {
      return *_logger.begin();
    } else {
      return nullptr;
    }
  }

  void bindLogger(std::shared_ptr<genie::log::Logger>& logger) { _logger.insert(logger); }

  QUALLA_API static std::shared_ptr<Env> create(const qualla::json& conf = {});
  QUALLA_API static std::shared_ptr<Env> create(std::istream& json_stream);
  QUALLA_API static std::shared_ptr<Env> create(const std::string& json_str);

 private:
  Path _path;
  std::unordered_set<std::shared_ptr<genie::log::Logger>> _logger;
};

}  // namespace qualla

#endif  // QUALLA_ENV_HPP
