//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <iostream>

#include "Exception.hpp"
#include "GenieSampler.h"
#include "Macro.hpp"
#include "Sampler.hpp"
#include "Util/HandleManager.hpp"
#include "qualla/detail/json.hpp"

using namespace genie;
GENIE_API
Genie_Status_t GenieSamplerConfig_createFromJson(const char* str,
                                                 GenieSamplerConfig_Handle_t* configHandle) {
  try {
    GENIE_ENSURE(str, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    auto config = std::make_shared<Sampler::Sampler::SamplerConfig>(str);
    GENIE_ENSURE(config, GENIE_STATUS_ERROR_MEM_ALLOC);
    *configHandle = Sampler::Sampler::SamplerConfig::add(config);
  } catch (const qualla::json::parse_error& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_JSON_FORMAT;
  } catch (const Exception& e) {
    std::cerr << e.what() << std::endl;
    return e.status();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieSamplerConfig_free(const GenieSamplerConfig_Handle_t configHandle) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    {
      // Check if the dialog actually exists
      auto configObj = Sampler::SamplerConfig::get(configHandle);
      GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    }
    Sampler::SamplerConfig::remove(configHandle);
  } catch (const std::exception& e) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieSamplerConfig_setParam(const GenieSamplerConfig_Handle_t configHandle,
                                           const char* keyStr,
                                           const char* valueStr) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto samplerConfig = Sampler::SamplerConfig::get(configHandle);
    GENIE_ENSURE(samplerConfig, GENIE_STATUS_ERROR_INVALID_HANDLE);
    samplerConfig->setParam(keyStr, valueStr);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_SET_PARAMS_FAILED;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieSampler_applyConfig(const GenieSampler_Handle_t samplerHandle,
                                        const GenieSamplerConfig_Handle_t configHandle) {
  try {
    GENIE_ENSURE(samplerHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);

    auto sampler = Sampler::get(samplerHandle);
    GENIE_ENSURE(sampler, GENIE_STATUS_ERROR_INVALID_HANDLE);

    auto samplerConfig = Sampler::SamplerConfig::get(configHandle);
    GENIE_ENSURE(samplerConfig, GENIE_STATUS_ERROR_INVALID_HANDLE);

    sampler->applyConfig(samplerConfig->getJson());

  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_APPLY_CONFIG_FAILED;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieSampler_registerCallback(const char* name,
                                             GenieSampler_ProcessCallback_t samplerCallback) {
  try {
    GENIE_ENSURE(samplerCallback, GENIE_STATUS_ERROR_INVALID_ARGUMENT);

    Sampler::registerCallback(name, samplerCallback);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieSampler_registerUserDataCallback(const char* name,
                                                     GenieSampler_UserDataCallback_t samplerCallback,
                                                     const void* userData) {
  try {
    GENIE_ENSURE(samplerCallback, GENIE_STATUS_ERROR_INVALID_ARGUMENT);

    Sampler::registerUserDataCallback(name, samplerCallback, userData);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}
