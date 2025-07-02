//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_VERSION_HPP
#define QUALLA_VERSION_HPP

#include <cstdint>

#include "qualla/detail/exports.h"

namespace qualla {

struct Version {
  QUALLA_API static int32_t major();
  QUALLA_API static int32_t minor();
  QUALLA_API static int32_t patch();
};

}  // namespace qualla

#endif  // QUALLA_VERSION_HPP
