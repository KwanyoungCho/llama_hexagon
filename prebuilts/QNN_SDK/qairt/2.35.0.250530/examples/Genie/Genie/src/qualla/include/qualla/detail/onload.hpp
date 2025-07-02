//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_DETAIL_ONLOAD_HPP
#define QUALLA_DETAIL_ONLOAD_HPP

#include <functional>

namespace qualla {

class OnLoad {
 public:
  OnLoad(std::function<void()> func) { func(); }
};

}  // namespace qualla

#endif  // QUALLA_DETAIL_ONLOAD_HPP
