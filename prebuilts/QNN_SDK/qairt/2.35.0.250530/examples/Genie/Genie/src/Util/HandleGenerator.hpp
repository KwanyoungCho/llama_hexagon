//==============================================================================
//
//  Copyright (c) 2019-2020,2023 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <mutex>

namespace qnn {
namespace util {

typedef std::size_t Handle_t;

class HandleGenerator final {
  static_assert(std::is_integral<Handle_t>::value, "Handle must be an integral type");
  static_assert((sizeof(Handle_t) == 8) || (sizeof(Handle_t) == 4),
                "Implementation of HandleGenerator::bswap() for sizeof(std::size_t) is required");

 public:
  HandleGenerator(const HandleGenerator&) = delete;
  HandleGenerator& operator=(const HandleGenerator&) = delete;
  HandleGenerator(HandleGenerator&&)                 = delete;
  HandleGenerator& operator=(HandleGenerator&&) = delete;

  static Handle_t generate(const void* const addr) {
    return (bswap((Handle_t)addr) ^ (Handle_t)s_operand);
  }
  static const void* reverse(const Handle_t handle) {
    return (void*)bswap(handle ^ (Handle_t)s_operand);
  }
  static constexpr Handle_t invalid() { return s_operand; }

 private:
// parasoft-instrumentation coverage off
  HandleGenerator() {}
// parasoft-instrumentation coverage on

  static uint32_t bswap32(const uint32_t val) {
    return (val >> 24U) | ((val >> 8U) & 0xff00U) | ((val << 8U) & 0xff0000U) | (val << 24U);
  }

  static uint64_t bswap64(const uint64_t val) {
    return ((bswap32(val) + 0ULL) << 32U) | bswap32(val >> 32U);
  }

  template <typename T>
  static size_t bswap(T val) {
    if (sizeof(T) == 4) {
      return bswap32(val);
    } else {
      return bswap64(val);
    }
  }

  // Magic number generated via "openssl rand -hex 8"
  static constexpr Handle_t s_operand = (Handle_t)0xd4c2416534bcdc9b;
};

}  // namespace util
}  // namespace qnn
