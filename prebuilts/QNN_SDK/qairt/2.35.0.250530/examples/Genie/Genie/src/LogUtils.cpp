//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <chrono>
#include <iostream>

#include "LogUtils.hpp"

#define MAX_LENGTH 1024

#ifdef __ANDROID__
void genie::log::utils::logLogcatCallback(const GenieLog_Handle_t handle,
                                          const char* fmt,
                                          GenieLog_Level_t level,
                                          uint64_t /* timestamp, will use logcat's */,
                                          va_list args) {
  android_LogPriority prio = ANDROID_LOG_DEFAULT;
  switch (level) {
    case GENIE_LOG_LEVEL_ERROR: {
      prio = ANDROID_LOG_ERROR;
      break;
    }
    case GENIE_LOG_LEVEL_WARN: {
      prio = ANDROID_LOG_WARN;
      break;
    }
    case GENIE_LOG_LEVEL_INFO: {
      prio = ANDROID_LOG_INFO;
      break;
    }
    case GENIE_LOG_LEVEL_VERBOSE: {
      prio = ANDROID_LOG_VERBOSE;
      break;
    }
    default: {
      prio = ANDROID_LOG_INFO;
      break;
    }
  }
  __android_log_vprint(prio, ANDROID_GENIE_LOG_TAG, fmt, args);
}

#else
uint64_t genie::log::utils::m_utilsEpoch = 0;

void genie::log::utils::setEpoch(const uint64_t epoch) { m_utilsEpoch = epoch; }

void genie::log::utils::populateLogString(char* const buf,
                                          const size_t bufSize,
                                          const char* const fmt,
                                          const GenieLog_Level_t level,
                                          const uint64_t timestamp,
                                          va_list argp) {
  const char* levelStr = "";
  switch (level) {
    case GENIE_LOG_LEVEL_ERROR:
      levelStr = " ERROR ";
      break;
    case GENIE_LOG_LEVEL_WARN:
      levelStr = "WARNING";
      break;
    case GENIE_LOG_LEVEL_INFO:
      levelStr = "  INFO ";
      break;
    case GENIE_LOG_LEVEL_VERBOSE:
      levelStr = "VERBOSE";
      break;
  }

  double ms = 0;

  // The backend is calling the callback directly instead of Logger::log with
  // zero value for timestamp. See LogHtpUtils.cpp for more info.
  if (timestamp == 0) {
    ms = (double)getHostTimestamp(m_utilsEpoch) / 1000000.0;
  } else {
    ms = (double)timestamp / 1000000.0;
  }

  int offset = snprintf(buf, bufSize, "\nGenie: %8.1fms [%-7s] ", ms, levelStr);
  vsnprintf(buf + offset, bufSize - (size_t)offset, fmt, argp);
}

#ifdef _WIN32
void genie::log::utils::registerETWProvider() { TraceLoggingRegister(genie_tracelogging_provider); }

void genie::log::utils::unregisterETWProvider() {
  TraceLoggingUnregister(genie_tracelogging_provider);
}

void genie::log::utils::logETWCallback(const GenieLog_Handle_t handle,
                                       const char* const fmt,
                                       const GenieLog_Level_t level,
                                       const uint64_t timestamp,
                                       va_list argp) {
  char str[MAX_LENGTH] = "";
  populateLogString(str, MAX_LENGTH, fmt, level, timestamp, argp);

  // TODO: When we no longer want to support printing logs to console,
  // the following printf should be removed.
  printf("%s\n", str);

  // Log to ETW.
  TraceLoggingWrite(genie_tracelogging_provider, "Genie", TraceLoggingValue(str, "TestMessage"));
}
#endif

void genie::log::utils::logStdoutCallback(const GenieLog_Handle_t handle,
                                          const char* const fmt,
                                          const GenieLog_Level_t level,
                                          const uint64_t timestamp,
                                          va_list argp) {
  char str[MAX_LENGTH] = "";

  populateLogString(str, MAX_LENGTH, fmt, level, timestamp, argp);
  printf("%s\n", str);
}
#endif

uint64_t genie::log::utils::getHostTimestamp(const uint64_t epoch) {
  return getTimestampSinceEpoch() - epoch;
}

uint64_t genie::log::utils::getTimestampSinceEpoch() {
  // add a static_cast to avoid a coding violation of implicit conversions
  return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(
                                   std::chrono::system_clock::now().time_since_epoch())
                                   .count());
}
