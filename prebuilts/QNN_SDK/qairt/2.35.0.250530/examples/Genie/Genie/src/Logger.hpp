//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef LOGGER_HPP
#define LOGGER_HPP

#include <atomic>
#include <cstdarg>
#include <iostream>
#include <memory>

#include "GenieLog.h"
#include "Util/HandleManager.hpp"

// To suppress token pasting warnings with gcc compiler
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
// To suppress token pasting warnings with clang compiler
#elif defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wpedantic"
#endif

// To make the logHandle as typedef to pointer to const object
// Existing logHandle is a typedef to pointer to object
typedef const void* Genie_Const_LogHandle_t;

#ifdef GENIE_ENABLE_DEBUG
#define GENIE_LOG_FILE __FILE__
#define GENIE_LOG_LINE __LINE__
#else
#define GENIE_LOG_FILE ""
#define GENIE_LOG_LINE 0
#endif

#define LOG(logHandle, level, fmt, ...)                                           \
  do {                                                                            \
    const auto logger = genie::log::Logger::getLogger(logHandle);                 \
    if (logger && (level) <= logger->getMaxLevel()) {                             \
      logger->log((level), GENIE_LOG_FILE, GENIE_LOG_LINE, (fmt), ##__VA_ARGS__); \
    }                                                                             \
  } while (0);

#define _LOG(logger, level, fmt, ...)                                                     \
  do {                                                                                    \
    if (logger && (level) <= logger->getMaxLevel()) {                                     \
      logger->log((level), GENIE_LOG_FILE, GENIE_LOG_LINE, (fmt).c_str(), ##__VA_ARGS__); \
    }                                                                                     \
  } while (0);

#define LOG2_ERROR(logHandle, fmt, ...) \
  LOG((logHandle), GENIE_LOG_LEVEL_ERROR, (fmt), ##__VA_ARGS__)
#define LOG2_WARNING(logHandle, fmt, ...) \
  LOG((logHandle), GENIE_LOG_LEVEL_WARN, (fmt), ##__VA_ARGS__)
#define LOG2_INFO(logHandle, fmt, ...) LOG((logHandle), GENIE_LOG_LEVEL_INFO, (fmt), ##__VA_ARGS__)
#define LOG2_VERBOSE(logHandle, fmt, ...) \
  LOG((logHandle), GENIE_LOG_LEVEL_VERBOSE, (fmt), ##__VA_ARGS__)

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#elif defined(__clang__)
#pragma clang diagnostic pop
#endif

namespace genie {
namespace log {

class Logger final {
 public:
  explicit Logger(GenieLog_Callback_t callback, const GenieLog_Level_t maxLevel, bool* status);

 public:
  Logger(const Logger&)            = delete;
  Logger& operator=(const Logger&) = delete;
  Logger(Logger&&)                 = delete;
  Logger& operator=(Logger&&)      = delete;

  bool setMaxLevel(const GenieLog_Level_t maxLevel);

  GenieLog_Level_t getMaxLevel() { return m_maxLevel; }

  GenieLog_Callback_t getCallback() { return m_callback; }

  const GenieLog_Handle_t getHandle() { return m_handle; }

  void log(const GenieLog_Level_t level, const char* file, const long line, const char* fmt, ...);

  void logByVaList(
      const GenieLog_Level_t level, const char* file, long line, const char* fmt, va_list args);

  void logByVaList(const GenieLog_Level_t level, const char* fmt, va_list argp);

  void incrementUseCount() { m_useCount++; }

  void decrementUseCount() { m_useCount--; }

  uint32_t getUseCount() { return m_useCount; }

  static GenieLog_Handle_t createLogger(GenieLog_Callback_t callback,
                                        GenieLog_Level_t maxLevel,
                                        bool* status);

  static bool isValid(const Genie_Const_LogHandle_t logHandle) {
    return (s_logManager.get((qnn::util::Handle_t)logHandle) != nullptr);
  }

  static void reset(const Genie_Const_LogHandle_t logHandle) {
    s_logManager.remove((qnn::util::Handle_t)logHandle);
  }
  static std::shared_ptr<Logger> getLogger(const Genie_Const_LogHandle_t logHandle = nullptr);

 private:
  void setHandle(GenieLog_Handle_t handle);

  GenieLog_Handle_t m_handle     = nullptr;
  GenieLog_Callback_t m_callback = nullptr;
  std::atomic<GenieLog_Level_t> m_maxLevel;
  uint64_t m_epoch;
  std::atomic<uint32_t> m_useCount{0};
  static qnn::util::HandleManager<Logger> s_logManager;
};

}  // namespace log
}  // namespace genie

#endif
