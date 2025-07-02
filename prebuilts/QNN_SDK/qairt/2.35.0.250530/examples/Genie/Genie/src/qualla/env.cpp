//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>

#include "qualla/env.hpp"

namespace fs = std::filesystem;

namespace qualla {

Env::Env(const json& conf) {
  _path.models = fs::path();
  _path.cache  = fs::path();

  if (conf.contains("path")) {
    const json& p = conf["path"];

    if (p.contains("models"))
      _path.models = fs::path(p["models"].get<std::string>()).make_preferred();
    if (p.contains("cache")) _path.cache = fs::path(p["cache"].get<std::string>()).make_preferred();
  }
}

Env::~Env() {}

std::shared_ptr<Env> Env::create(const qualla::json& conf) { return std::make_shared<Env>(conf); }

std::shared_ptr<Env> Env::create(std::istream& json_stream) {
  return create(json::parse(json_stream));
}

std::shared_ptr<Env> Env::create(const std::string& json_str) {
  return create(json::parse(json_str));
}

}  // namespace qualla
