//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include "Exception.hpp"
#include "GenieProfile.h"
#include "Macro.hpp"
#include "Profile.hpp"

using namespace genie;

#ifdef __cplusplus
extern "C" {
#endif

GENIE_API
Genie_Status_t GenieProfile_create(const GenieProfileConfig_Handle_t configHandle,
                                   GenieProfile_Handle_t* profileHandle) {
  try {
    GENIE_ENSURE(profileHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    // Config Handle must be NULL per the current API definition.
    GENIE_ENSURE(!configHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    const std::shared_ptr<Profiler> profile(new Profiler());
    GENIE_ENSURE(profile, GENIE_STATUS_ERROR_MEM_ALLOC);
    profile->setLevel(GENIE_PROFILE_LEVEL_BASIC);
    *profileHandle = Profiler::add(profile);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieProfile_getJsonData(const GenieProfile_Handle_t profileHandle,
                                        Genie_AllocCallback_t callback,
                                        const char** jsonData) {
  try {
    GENIE_ENSURE(profileHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(callback, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(jsonData, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    const std::shared_ptr<Profiler> profile = Profiler::get(profileHandle);
    GENIE_ENSURE(profile, GENIE_STATUS_ERROR_INVALID_HANDLE);
    const uint32_t jsonSize = profile->serialize();
    callback(jsonSize, jsonData);
    profile->getJsonData(jsonData);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieProfile_free(GenieProfile_Handle_t profileHandle) {
  try {
    GENIE_ENSURE(profileHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    const std::shared_ptr<Profiler> profile = Profiler::get(profileHandle);
    GENIE_ENSURE(profile, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(!profile->getUseCount(), GENIE_STATUS_ERROR_BOUND_HANDLE);
    Profiler::remove(profileHandle);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif
