//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>

#include "qualla/detail/kpi.hpp"

namespace qualla {

std::string Kpi::dump(std::string_view sep) const {
  return fmt::format(
      "last:{:.2f}{}total:{:.2f}{}min:{:.2f}{}max:{:.2f}{}avg:{:.2f} (msec){}count:{}",
      last_usec / 1000.0,
      sep,
      total_usec / 1000.0,
      sep,
      min_usec / 1000.0,
      sep,
      max_usec / 1000.0,
      sep,
      total_usec / (count ? count : 1) / 1000.0,
      sep,
      count);
}

}  // namespace qualla
