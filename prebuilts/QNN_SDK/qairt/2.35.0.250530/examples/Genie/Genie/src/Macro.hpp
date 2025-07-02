//============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#pragma once

#define ENABLE_DEBUG_LOGS 0

//======================================================================================================================
// Error generation macros
//======================================================================================================================

#define GENIE_LOG_ERROR(fmt, ...)

#define GENIE_ENSURE_MSG(value, return_error, msg) \
  do {                                             \
    if (!(value)) {                                \
      GENIE_LOG_ERROR(" " msg);                    \
      return return_error;                         \
    }                                              \
  } while (0)

#define GENIE_ENSURE(value, return_error)          \
  do {                                             \
    if (!(value)) {                                \
      GENIE_LOG_ERROR("%s was not true.", #value); \
      return return_error;                         \
    }                                              \
  } while (0)

#define GENIE_ENSURE_STATUS(status, return_error) \
  do {                                            \
    if ((status) != GENIE_SUCCESS) {              \
      return return_error;                        \
    }                                             \
  } while (0)

#define GENIE_ENSURE_EQ(a, b, return_error)                     \
  do {                                                          \
    if ((a) != (b)) {                                           \
      GENIE_LOG_ERROR("%s != %s (%d != %d)", #a, #b, (a), (b)); \
      return return_error;                                      \
    }                                                           \
  } while (0)

#define GENIE_ENSURE_NOT_EMPTY(value, return_error) \
  do {                                              \
    if (value.empty()) {                            \
      GENIE_LOG_ERROR("%s was not true.", #value);  \
      return return_error;                          \
    }                                               \
  } while (0)
//======================================================================================================================
// JSON config macros
//======================================================================================================================

#define JSON_ENFORCE_OBJECT()                                                                 \
  if (!item.value().is_object()) {                                                            \
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,                                           \
                    "Invalid " + component + " config: " + item.key() + " is not an object"); \
  }

#define JSON_ENFORCE_ARRAY()                                                                 \
  if (!item.value().is_array()) {                                                            \
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,                                          \
                    "Invalid " + component + " config: " + item.key() + " is not an array"); \
  }

#define JSON_ENFORCE_ARRAY_OR_OBJECT()                                                     \
  if (!item.value().is_array() && !item.value().is_object()) {                             \
    throw Exception(                                                                       \
        GENIE_STATUS_ERROR_JSON_SCHEMA,                                                    \
        "Invalid " + component + " config: " + item.key() + " is not an array or object"); \
  }

#define JSON_ENFORCE_NUMERIC()                                                              \
  if (!item.value().is_number()) {                                                          \
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,                                         \
                    "Invalid " + component + " config: " + item.key() + " is not numeric"); \
  }

#define JSON_ENFORCE_ARRAY_OR_NUMERIC()                                                     \
  if (!item.value().is_number() && !item.value().is_array()) {                              \
    throw Exception(                                                                        \
        GENIE_STATUS_ERROR_JSON_SCHEMA,                                                     \
        "Invalid " + component + " config: " + item.key() + " is not an array or numeric"); \
  }

#define JSON_ENFORCE_BOOLEAN()                                                              \
  if (!item.value().is_boolean()) {                                                         \
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,                                         \
                    "Invalid " + component + " config: " + item.key() + " is not boolean"); \
  }

#define JSON_ENFORCE_STRING()                                                                \
  if (!item.value().is_string()) {                                                           \
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,                                          \
                    "Invalid " + component + " config: " + item.key() + " is not a string"); \
  }
