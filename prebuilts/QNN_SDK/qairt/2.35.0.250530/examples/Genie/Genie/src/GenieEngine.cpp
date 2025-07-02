//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <iostream>

#include "Engine.hpp"
#include "Exception.hpp"
#include "GenieEngine.h"
#include "Macro.hpp"
#include "Util/HandleManager.hpp"
#include "qualla/detail/json.hpp"

using namespace genie;
GENIE_API
Genie_Status_t GenieEngineConfig_createFromJson(const char* str,
                                                GenieEngineConfig_Handle_t* configHandle) {
  try {
    GENIE_ENSURE(str, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    auto config = std::make_shared<Engine::EngineConfig>(str);
    GENIE_ENSURE(config, GENIE_STATUS_ERROR_MEM_ALLOC);
    *configHandle = Engine::EngineConfig::add(config);
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
Genie_Status_t GenieEngineConfig_free(const GenieEngineConfig_Handle_t configHandle) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    {
      // Check if the engine config actually exists
      auto configObj = Engine::EngineConfig::get(configHandle);
      GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    }
    Engine::EngineConfig::remove(configHandle);
  } catch (const std::exception& e) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieEngine_create(const GenieEngineConfig_Handle_t configHandle,
                                  GenieEngine_Handle_t* engineHandle) {
  try {
    GENIE_ENSURE(engineHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    // Get config object
    auto configObj = Engine::EngineConfig::get(configHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    // Create engine
    auto engine = std::make_shared<genie::Engine>(configObj);
    GENIE_ENSURE(engine, GENIE_STATUS_ERROR_MEM_ALLOC);
    // Create Handle
    *engineHandle = genie::Engine::add(engine);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  // Return SUCCESS
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieEngine_free(const GenieEngine_Handle_t engineHandle) {
  try {
    GENIE_ENSURE(engineHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    genie::Engine::remove(engineHandle);
  } catch (const std::exception& e) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}