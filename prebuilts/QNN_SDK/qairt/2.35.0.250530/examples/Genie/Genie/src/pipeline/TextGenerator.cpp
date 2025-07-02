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
#include "TextGenerator.hpp"

using namespace genie;

pipeline::TextGenerator::TextGenerator(qualla::json config,
                                       std::shared_ptr<ProfileStat> profileStat,
                                       std::shared_ptr<genie::log::Logger> logger)
    : Node(config) {
  m_typeGenerator = true;
  for (auto item : m_config.items()) {
    Dialog::validateDialogConfig(item.value());
    // Create dialog
    qualla::json dialogConfig;
    dialogConfig["dialog"] = item.value();
    if (dialogConfig["dialog"].contains("accumulator-size")) {
      m_accumulatorSize = item.value()["accumulator-size"];
    }
    m_generator = std::make_shared<genie::Dialog>(dialogConfig, profileStat, logger);
  }
}

int32_t pipeline::TextGenerator::bindPipeline(Pipeline& pipeline) {
  if (m_pipeline) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "Node already bound to Pipeline");
  }
  m_pipeline = &pipeline;
  m_pipeline->setupAccumultator(m_accumulatorSize);
  std::string inputDataType = "QNN_DATATYPE_FLOAT_32";
  double inputScale         = 1.0;
  int32_t inputOffset       = 0;
  size_t inputByteWidth     = 4;
  m_generator->getInputQuantParam(inputDataType, inputScale, inputOffset, inputByteWidth);
  // Set encoding for accumulator
  m_pipeline->m_accumulator->setEncoding(inputDataType, inputScale, inputOffset, inputByteWidth);
  return GENIE_STATUS_SUCCESS;
}

// Appends query string to existing query string
int32_t pipeline::TextGenerator::setTextInputData(GenieNode_IOName_t nodeIOName, const char* txt) {
  if (nodeIOName != GenieNode_IOName_t::GENIE_NODE_TEXT_GENERATOR_TEXT_INPUT) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL,
                    "setTextInputData can only be set for GENIE_NODE_TEXT_GENERATOR_TEXT_INPUT");
  }
  m_queryString += txt;
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::TextGenerator::setEmbeddingInputData(GenieNode_IOName_t nodeIOName,
                                                       const void* embedding,
                                                       size_t embeddingSize) {
  if (nodeIOName != GenieNode_IOName_t::GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT) {
    throw Exception(
        GENIE_STATUS_ERROR_GENERAL,
        "setTextInputData can only be set for GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT");
  }
  m_pipeline->m_accumulator->append((uint8_t*)embedding, embeddingSize);
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::TextGenerator::setTextOutputCallback(GenieNode_IOName_t nodeIOName,
                                                       GenieNode_TextOutput_Callback_t callback) {
  m_textOutputCallback = callback;
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::TextGenerator::execute(void* userData) {
  if (m_pipeline->m_accumulator->getDataSize()) {
    m_generator->embeddingQuery(m_pipeline->m_accumulator->getData(),
                                m_pipeline->m_accumulator->getDataSize(),
                                GenieNode_TextOutput_SentenceCode_t::GENIE_NODE_SENTENCE_COMPLETE,
                                m_textOutputCallback,
                                userData,
                                nullptr);
  } else {
    m_generator->query(m_queryString.c_str(),
                       GenieNode_TextOutput_SentenceCode_t::GENIE_NODE_SENTENCE_COMPLETE,
                       m_textOutputCallback,
                       userData,
                       nullptr);
  }
  m_pipeline->m_accumulator->flush();
  m_queryString.clear();
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::TextGenerator::save(const std::string& name) { return m_generator->save(name); }

int32_t pipeline::TextGenerator::restore(const std::string& name) {
  return m_generator->restore(name);
}

void pipeline::TextGenerator::reset() { m_generator->reset(); }

int32_t pipeline::TextGenerator::setPriority(std::string engine,
                                             const GeniePipeline_Priority_t priority) {
  return m_generator->setPriority(engine, (GenieDialog_Priority_t)priority);
}

int32_t pipeline::TextGenerator::setOemkey(const std::string& oemKey) {
  return m_generator->setOemkey(oemKey);
}

int32_t pipeline::TextGenerator::applyLora(std::string loraAdapterName, std::string engine) {
  auto status = m_generator->applyLora(loraAdapterName, engine);
  if (m_pipeline) {  // Node configured to pipeline
    std::string inputDataType = "QNN_DATATYPE_FLOAT_32";
    double inputScale         = 1.0;
    int32_t inputOffset       = 0;
    size_t inputByteWidth     = 4;
    m_generator->getInputQuantParam(inputDataType, inputScale, inputOffset, inputByteWidth);
    // Set encoding for accumulator
    m_pipeline->m_accumulator->flush();
    m_pipeline->m_accumulator->setEncoding(inputDataType, inputScale, inputOffset, inputByteWidth);
  }
  return status;
}

int32_t pipeline::TextGenerator::applyLoraStrength(std::string tensorName,
                                                   std::string engine,
                                                   float alpha) {
  return m_generator->applyLoraStrength(tensorName, engine, alpha);
}

GenieEngine_Handle_t pipeline::TextGenerator::getEngineHandle(const std::string& engineRole) {
  return m_generator->getEngineHandle(engineRole);
}

int32_t pipeline::TextGenerator::bindEngine(const std::string& engineRole,
                                            std::shared_ptr<Engine> engine) {
  return m_generator->bindEngine(engineRole, engine);
}

GenieSampler_Handle_t pipeline::TextGenerator::getSamplerHandle() {
  return genie::Dialog::getSamplerHandle(m_generator);
}

GenieTokenizer_Handle_t pipeline::TextGenerator::getTokenizerHandle() {
  return genie::Dialog::getTokenizerHandle(m_generator);
}