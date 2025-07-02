//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <string>
#include <vector>

#include "qualla/detail/config.hpp"
#include "qualla/detail/exports.h"

namespace qualla {

class GpioMarker {
 public:
  QUALLA_API GpioMarker(const qualla::json& conf);
  QUALLA_API virtual ~GpioMarker();

  // Perform a pull-up or pull-down operation on GPIO
  QUALLA_API virtual void set();

  // Set GPIO to low
  QUALLA_API virtual void reset();

  QUALLA_API static std::unique_ptr<GpioMarker> create(std::istream& json_stream);
  QUALLA_API static std::unique_ptr<GpioMarker> create(const std::string& json_str);
  QUALLA_API static std::unique_ptr<GpioMarker> create(const qualla::json& conf = {});

 private:
  std::string _tool_path;       // Gpio-set tool path
  std::string _command;         // Command for gpio-set tool
  int32_t _gpio_num;            // Gpio num to be set
  int32_t _gpio_status;         // Gpio current status(pull up or pull down)
  int32_t _gpio_marker_enable;  // Flag of function enable or not
};

}  // namespace qualla
