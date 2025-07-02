//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

/**
 * @brief A handle for engine instances.
 */

#ifndef GENIE_ENGINE_H
#define GENIE_ENGINE_H

#include "GenieCommon.h"
#ifdef __cplusplus
extern "C" {
#endif

//=============================================================================
// Data Types
//=============================================================================

/**
 * @brief A handle for engine configuration instances.
 */
typedef const struct _GenieEngineConfig_Handle_t* GenieEngineConfig_Handle_t;

/**
 * @brief A handle for engine instances.
 */
typedef const struct _GenieEngine_Handle_t* GenieEngine_Handle_t;

//=============================================================================
// Functions
//=============================================================================

/**
 * @brief A function to create a engine configuration from a JSON string.
 *
 * @param[in] str A configuration string. Must not be NULL.
 *
 * @param[out] configHandle A handle to the created engine config. Must not be NULL.
 *
 * @return Status code:
 *         - GENIE_STATUS_SUCCESS: API call was successful.
 *         - GENIE_STATUS_ERROR_INVALID_ARGUMENT: At least one argument is invalid.
 *         - GENIE_STATUS_ERROR_MEM_ALLOC: Memory allocation failure.
 *         - GENIE_STATUS_ERROR_INVALID_CONFIG: At least one configuration option is invalid.
 */
GENIE_API
Genie_Status_t GenieEngineConfig_createFromJson(const char* str,
                                                GenieEngineConfig_Handle_t* configHandle);

/**
 * @brief A function to free a engine config.
 *
 * @param[in] configHandle A engine config handle.
 *
 * @return Status code:
 *         - GENIE_STATUS_SUCCESS: API call was successful.
 *         - GENIE_STATUS_ERROR_INVALID_HANDLE: Engine handle is invalid.
 *         - GENIE_STATUS_ERROR_MEM_ALLOC: Memory (de)allocation failure.
 */
GENIE_API
Genie_Status_t GenieEngineConfig_free(const GenieEngineConfig_Handle_t configHandle);

/**
 * @brief A function to create a handle to a engine object.
 *
 * @param[in] configHandle A engine configuration options. Must be NULL.
 *
 * @param[out] engineHandle A handle to the created engine handle. Must not be NULL.
 *
 * @return Status code:
 *         - GENIE_STATUS_SUCCESS: API call was successful.
 *         - GENIE_STATUS_ERROR_INVALID_ARGUMENT: At least one argument is invalid.
 *         - GENIE_STATUS_ERROR_MEM_ALLOC: Memory allocation failure.
 */
GENIE_API
Genie_Status_t GenieEngine_create(const GenieEngineConfig_Handle_t configHandle,
                                  GenieEngine_Handle_t* engineHandle);

/**
 * @brief A function to free memory associated with a engine handle. This call
 *        will fail if the engine handle is still bound to a active dialog.
 *
 * @param[in] engineHandle A engine handle. Must not be NULL
 *
 * @return Status code:
 *         - GENIE_STATUS_SUCCESS: API call was successful.
 *         - GENIE_STATUS_ERROR_INVALID_HANDLE: Engine handle is invalid.
 *         - GENIE_STATUS_ERROR_BOUND_HANDLE: Engine handle is bound to another handle.
 *         - GENIE_STATUS_ERROR_MEM_ALLOC: Memory (de)allocation failure.
 */
GENIE_API
Genie_Status_t GenieEngine_free(const GenieEngine_Handle_t engineHandle);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // GENIE_ENGINE_H
