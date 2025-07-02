//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_CLIP_IMAGE_PROCESSOR_HPP
#define QUALLA_CLIP_IMAGE_PROCESSOR_HPP

#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "qualla/detail/exports.h"
#include "qualla/encoder.hpp"
#include "qualla/engine.hpp"
#include "qualla/env.hpp"

namespace qualla {

class ImageEncoder : public Encoder {
 public:
  ImageEncoder(std::shared_ptr<Env> env, const qualla::json& conf);
  virtual ~ImageEncoder();

  virtual bool encode(std::vector<uint8_t>& pixel_values, std::vector<uint8_t>& image_features);

  // Get output dimensions
  void output_dimensions(std::vector<std::uint32_t>& outputDimensions);

  void outputTensorQuantParam(std::string& dataType,
                              double& scale,
                              int32_t& offset,
                              size_t& byteWidth);

 protected:
  std::vector<std::uint32_t> _output_dimensions{};

  virtual bool process(std::vector<uint8_t>& pixel_values, std::vector<uint8_t>& image_features);

  size_t _model_input_height    = 384;
  size_t _model_input_width     = 384;
  size_t _model_input_channel   = 3;
  size_t _model_input_byteWidth = 1;
};

}  // namespace qualla

#endif  // QUALLA_DIALOG_HPP
