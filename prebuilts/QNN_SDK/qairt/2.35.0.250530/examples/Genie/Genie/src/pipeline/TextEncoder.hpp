//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once
#ifndef TEXT_ENCODER_HPP
#define TEXT_ENCODER_HPP
#include <memory>
#include <vector>

#include "GeniePipeline.h"
#include "Node.hpp"

namespace genie {
namespace pipeline {
class Pipeline;
class TextEncoder : public Node {
 public:
  TextEncoder(qualla::json config,
              std::shared_ptr<ProfileStat> profileStat,
              std::shared_ptr<genie::log::Logger> logger = nullptr);
  ~TextEncoder();
  int32_t execute(void* userData);
  int32_t setTextInputData(GenieNode_IOName_t nodeIOName, const char* txt);
  int32_t setEmbeddingOutputCallback(GenieNode_IOName_t nodeIOName,
                                     GenieNode_EmbeddingOutputCallback_t callback);

  int32_t applyLora(std::string loraAdapterName, std::string engine);
  int32_t applyLoraStrength(std::string tensorName, std::string engine, float alpha);

 private:
  std::shared_ptr<Embedding> m_encoder;
  std::string m_type = "lut";
  std::vector<uint8_t> m_data;
  GenieNode_EmbeddingOutputCallback_t m_embeddingOutputCallback;
};
}  // namespace pipeline
}  // namespace genie
#endif  // TEXT_ENCODER_HPP