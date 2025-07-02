//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <memory>
#include <vector>

#include "Exception.hpp"
#include "Pipeline.hpp"
#include "TextEncoder.hpp"

using namespace genie;

pipeline::TextEncoder::TextEncoder(qualla::json config,
                                   std::shared_ptr<ProfileStat> profileStat,
                                   std::shared_ptr<genie::log::Logger> logger)
    : Node(config) {
  for (auto item : m_config.items()) {
    Embedding::validateEmbeddingConfig(item.value(), false);
    qualla::json embeddingConfig;
    embeddingConfig["embedding"] = item.value();
    m_encoder = std::make_shared<genie::Embedding>(embeddingConfig, profileStat, logger);
  }
}

int32_t pipeline::TextEncoder::setEmbeddingOutputCallback(
    GenieNode_IOName_t nodeIOName, GenieNode_EmbeddingOutputCallback_t callback) {
  if (nodeIOName != GenieNode_IOName_t::GENIE_NODE_TEXT_ENCODER_EMBEDDING_OUTPUT) {
    throw Exception(
        GENIE_STATUS_ERROR_GENERAL,
        "setEmbeddingOutputCallback can only be set for GENIE_NODE_TEXT_ENCODER_EMBEDDING_OUTPUT");
  }
  m_embeddingOutputCallback = callback;
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::TextEncoder::setTextInputData(GenieNode_IOName_t nodeIOName, const char* txt) {
  if (nodeIOName != GenieNode_IOName_t::GENIE_NODE_TEXT_ENCODER_TEXT_INPUT) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL,
                    "setTextInputData can only be set for GENIE_NODE_TEXT_ENCODER_TEXT_INPUT");
  }
  m_encoder->encode(txt, m_data, nullptr);
  if (isConnected()) {
    std::string outputDataType = "QNN_DATATYPE_FLOAT_32";
    double outputScale         = 1.0;
    int32_t outputOffset       = 0;
    size_t outputByteWidth     = 4;
    m_encoder->getOutputQuantParam(outputDataType, outputScale, outputOffset, outputByteWidth);
    size_t numElements = m_data.size() / outputByteWidth;

    m_pipeline->m_accumulator->append(
        m_data.data(), outputDataType, outputScale, outputOffset, numElements);
  }
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::TextEncoder::execute(void* userData) {
  std::vector<uint32_t> dimensions;
  m_encoder->getOutputDimensions(dimensions);
  if (m_embeddingOutputCallback) {  // invoke userCallback if set
    m_embeddingOutputCallback(
        dimensions.data(), dimensions.size(), m_data.size(), (void*)m_data.data(), userData);
  }
  m_data.clear();  // clear encoder buffer after callback invoked
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::TextEncoder::applyLora(std::string loraAdapterName, std::string engine) {
  return m_encoder->applyLora(loraAdapterName, engine);
}

int32_t pipeline::TextEncoder::applyLoraStrength(std::string tensorName,
                                                 std::string engine,
                                                 float alpha) {
  return m_encoder->applyLoraStrength(tensorName, engine, alpha);
}

pipeline::TextEncoder::~TextEncoder() {}