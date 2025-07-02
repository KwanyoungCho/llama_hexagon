//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <cstdarg>
#include <cstdio>

#include "GenieLog.h"

#ifdef __ANDROID__
#include <android/log.h>
#define ANDROID_GENIE_LOG_TAG "Genie"
#endif

#ifdef _WIN32
// clang-format off
#include <windows.h>
#include <TraceLoggingProvider.h>
// clang-format on
TRACELOGGING_DECLARE_PROVIDER(genie_tracelogging_provider);
#endif

namespace genie {
namespace log {
namespace utils {
#ifdef __ANDROID__
void logLogcatCallback(const GenieLog_Handle_t handle,
                       const char* fmt,
                       GenieLog_Level_t level,
                       uint64_t /* timestamp, will use logcat's */,
                       va_list args);
#else
extern uint64_t m_utilsEpoch;
void setEpoch(const uint64_t epoch);
void populateLogString(char* const buf,
                       const size_t bufSize,
                       const char* const fmt,
                       const GenieLog_Level_t level,
                       const uint64_t timestamp,
                       va_list argp);
#ifdef _WIN32
void registerETWProvider();
void unregisterETWProvider();
void logETWCallback(const GenieLog_Handle_t handle,
                    const char* fmt,
                    GenieLog_Level_t level,
                    uint64_t timestamp,
                    va_list argp);
#endif
void logStdoutCallback(const GenieLog_Handle_t handle,
                       const char* fmt,
                       GenieLog_Level_t level,
                       uint64_t timestamp,
                       va_list argp);
#endif

uint64_t getHostTimestamp(const uint64_t epoch);
uint64_t getTimestampSinceEpoch();

}  // namespace utils
}  // namespace log
}  // namespace genie
