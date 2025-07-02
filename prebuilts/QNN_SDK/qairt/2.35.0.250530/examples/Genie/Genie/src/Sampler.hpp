//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once
#include <memory>

#include "GenieSampler.h"
#include "Util/HandleManager.hpp"
#include "qualla/env.hpp"
#include "qualla/sampler.hpp"

namespace genie {
class Sampler {
 public:
  class SamplerConfig {
   public:
    static GenieSamplerConfig_Handle_t add(std::shared_ptr<SamplerConfig> config);

    static std::shared_ptr<SamplerConfig> get(GenieSamplerConfig_Handle_t handle);

    static void remove(GenieSamplerConfig_Handle_t handle);

    static void validateSamplerConfig(const qualla::json& config);

    static void translateSamplerConfig(const qualla::json& genieConfig, qualla::json& quallaConfig);

    SamplerConfig(const char* configStr);

    void setParam(const std::string& keyStr, const std::string& valueStr);

    qualla::json getJson() const;

   private:
    static qnn::util::HandleManager<SamplerConfig> s_manager;
    qualla::json m_config;
  };

  static GenieSampler_Handle_t add(std::shared_ptr<Sampler> sampler);
  static std::shared_ptr<Sampler> get(GenieSampler_Handle_t handle);
  static void remove(GenieSampler_Handle_t handle);
  static void registerCallback(const char* name, GenieSampler_ProcessCallback_t samplerCallback);
  static void registerUserDataCallback(const char* name, GenieSampler_UserDataCallback_t samplerCallback, const void* userData);

  Sampler(qualla::json& origJson,
          std::vector<std::reference_wrapper<qualla::Sampler>>& quallaSamplers);

  void applyConfig(qualla::json samplerConfigJson);

  const qualla::json& getJson();

 private:
  qualla::json m_origJson;
  static qnn::util::HandleManager<Sampler> s_manager;
  std::vector<std::reference_wrapper<qualla::Sampler>> m_quallaSamplers;
};

}  // namespace genie
