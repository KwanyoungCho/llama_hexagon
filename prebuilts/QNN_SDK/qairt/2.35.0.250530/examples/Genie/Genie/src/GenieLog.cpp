//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include <iostream>

#include "Exception.hpp"
#include "GenieLog.h"
#include "LogUtils.hpp"
#include "Logger.hpp"
#include "Macro.hpp"

using namespace genie;

#ifdef __cplusplus
extern "C" {
#endif

GENIE_API
Genie_Status_t GenieLog_create(const GenieLogConfig_Handle_t configHandle,
                               const GenieLog_Callback_t callback,
                               const GenieLog_Level_t logLevel,
                               GenieLog_Handle_t* logHandle) {
  try {
    GENIE_ENSURE(configHandle == NULL, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(logHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    switch (logLevel) {
      case GENIE_LOG_LEVEL_ERROR:
      case GENIE_LOG_LEVEL_WARN:
      case GENIE_LOG_LEVEL_INFO:
      case GENIE_LOG_LEVEL_VERBOSE:
        // Do nothing
        break;
      default:
        return GENIE_STATUS_ERROR_INVALID_ARGUMENT;
    }
    bool status;
    *logHandle = genie::log::Logger::createLogger(callback, logLevel, &status);
    if (!status) return GENIE_STATUS_ERROR_GENERAL;
    LOG2_INFO(*logHandle, "Genie Logger created with level : %d", logLevel);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieLog_free(GenieLog_Handle_t logHandle) {
  try {
    GENIE_ENSURE(genie::log::Logger::isValid(logHandle), GENIE_STATUS_ERROR_INVALID_HANDLE);
    genie::log::Logger::reset(logHandle);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif
