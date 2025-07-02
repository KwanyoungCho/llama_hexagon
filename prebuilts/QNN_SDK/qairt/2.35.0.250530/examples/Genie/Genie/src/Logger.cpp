//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <cstdio>
#include <iostream>
#include <sstream>
#include <string>

#include "LogUtils.hpp"
#include "Logger.hpp"

#ifdef _WIN32
TRACELOGGING_DEFINE_PROVIDER(
    genie_tracelogging_provider,
    "GENIETraceLoggingProvider",
    (0xd1418f8b, 0x08de, 0x4e5d, 0xab, 0xde, 0x3a, 0xe0, 0xca, 0x5c, 0xcb, 0x82));
#endif

#define LOG_RETURN_STATUS_PTR(status, errCode) \
  do {                                         \
    if (status) {                              \
      *(status) = (errCode);                   \
    }                                          \
    return;                                    \
  } while (0);

using namespace genie::log;

qnn::util::HandleManager<Logger> Logger::s_logManager;
bool Logger::setMaxLevel(const GenieLog_Level_t maxLevel) {
  if ((maxLevel < GENIE_LOG_LEVEL_ERROR) || (maxLevel > GENIE_LOG_LEVEL_VERBOSE)) {
    return false;
  }
  m_maxLevel = maxLevel;
  return true;
}

GenieLog_Handle_t Logger::createLogger(GenieLog_Callback_t callback,
                                       GenieLog_Level_t maxLevel,
                                       bool* status) {
  const auto logger =
      std::shared_ptr<Logger>(new (std::nothrow) Logger(callback, maxLevel, status));
  if (!*status) {
    return nullptr;
  }
  GenieLog_Handle_t handle = (GenieLog_Handle_t)s_logManager.add(logger);
  logger->setHandle(handle);
  return handle;
}

Logger::Logger(GenieLog_Callback_t callback, const GenieLog_Level_t maxLevel, bool* const status)
    : m_callback(callback), m_maxLevel(maxLevel), m_epoch(utils::getTimestampSinceEpoch()) {
  if (!callback) {
#ifdef __ANDROID__
    m_callback = utils::logLogcatCallback;
#else
    utils::setEpoch(m_epoch);
#ifdef _WIN32
    m_callback = utils::logETWCallback;
#else
    m_callback = utils::logStdoutCallback;
#endif
#endif
  }

  if (status) {
    if ((maxLevel > GENIE_LOG_LEVEL_VERBOSE) || (maxLevel < GENIE_LOG_LEVEL_ERROR)) {
      *status = false;
    } else {
      *status = true;
    }
  }
}

void Logger::setHandle(GenieLog_Handle_t handle) { m_handle = handle; }

std::shared_ptr<Logger> Logger::getLogger(const Genie_Const_LogHandle_t logHandle) {
  return (s_logManager.get((qnn::util::Handle_t)logHandle));
}

void Logger::log(const GenieLog_Level_t level,
                 const char* const file,
                 const long line,
                 const char* const fmt,
                 ...) {
  if (m_callback) {
    auto ml = m_maxLevel.load(std::memory_order_seq_cst);
    if (level > ml) {
      return;
    }
    va_list argp;
    va_start(argp, fmt);
    logByVaList(level, file, line, fmt, argp);
    va_end(argp);
  }
}

void Logger::logByVaList(const GenieLog_Level_t level,
                         const char* const file,
                         long line,
                         const char* const fmt,
                         va_list argp) {
#ifdef GENIE_ENABLE_DEBUG
  // Append filename and line numbers for debug purposes.
  // Note that the use of stringstream objects increases library size
  std::ostringstream logString;
  logString << file << "[" << line << "]: " << fmt;
  (*m_callback)(logString.str().c_str(), level, utils::getHostTimestamp(m_epoch), argp);
#else
  const std::string logString(fmt);
  std::ignore = file;
  std::ignore = line;
  (*m_callback)(m_handle, logString.c_str(), level, utils::getHostTimestamp(m_epoch), argp);
#endif
}

void Logger::logByVaList(const GenieLog_Level_t level, const char* const fmt, va_list argp) {
  const std::string logString(fmt);
  (*m_callback)(m_handle, logString.c_str(), level, utils::getHostTimestamp(m_epoch), argp);
}
