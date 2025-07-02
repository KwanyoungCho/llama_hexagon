//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <memory>
#include <set>
#include <sstream>

#include "Exception.hpp"
#include "ImageEncoder.hpp"
#include "Macro.hpp"
#include "Node.hpp"
#include "Pipeline.hpp"
#include "TextEncoder.hpp"
#include "TextGenerator.hpp"

using namespace genie;

qnn::util::HandleManager<pipeline::Node::Config> pipeline::Node::Config::s_manager;

GenieNodeConfig_Handle_t pipeline::Node::Config::add(std::shared_ptr<Config> config) {
  return (GenieNodeConfig_Handle_t)s_manager.add(config);
}

std::shared_ptr<pipeline::Node::Config> pipeline::Node::Config::get(
    GenieNodeConfig_Handle_t handle) {
  return s_manager.get((qnn::util::Handle_t)handle);
}

void pipeline::Node::Config::remove(GenieNodeConfig_Handle_t handle) {
  s_manager.remove((qnn::util::Handle_t)handle);
}

pipeline::Node::Config::Config(const char* configStr) { m_config = qualla::json::parse(configStr); }

void pipeline::Node::Config::bindLogger(std::shared_ptr<genie::log::Logger> logger) {
  if (!logger) return;
  logger->incrementUseCount();
  m_logger.insert(logger);
}

void pipeline::Node::Config::unbindLogger() {
  for (auto it : m_logger) it->decrementUseCount();
  m_logger.clear();
}

std::unordered_set<std::shared_ptr<genie::log::Logger>>& pipeline::Node::Config::getLogger() {
  return m_logger;
}

void pipeline::Node::Config::bindProfiler(std::shared_ptr<Profiler> profiler) {
  if (!profiler) return;
  profiler->incrementUseCount();
  m_profiler.insert(profiler);
}

void pipeline::Node::Config::unbindProfiler() {
  for (auto it : m_profiler) it->decrementUseCount();
  m_profiler.clear();
}

std::unordered_set<std::shared_ptr<Profiler>>& pipeline::Node::Config::getProfiler() {
  return m_profiler;
}

//=============================================================================
// Node functions
//=============================================================================
qnn::util::HandleManager<pipeline::Node> pipeline::Node::s_manager;
std::atomic<std::uint32_t> pipeline::Node::s_nameCounter{0u};

std::shared_ptr<pipeline::Node> pipeline::Node::createNode(
    std::shared_ptr<Config> configObj,
    std::shared_ptr<ProfileStat> profileStat,
    std::shared_ptr<genie::log::Logger> logger) {
  auto config = configObj->getJson();
  // component is used in the "ENFORCE" macros
  std::string component = "node";
  // Translate pipeline-config to Dialog/ Embedding config
  for (auto item : config.items()) {
    if (item.key() == "lut-encoder") {
      JSON_ENFORCE_OBJECT();
      return std::make_shared<pipeline::TextEncoder>(item, profileStat, logger);
    } else if (item.key() == "text-encoder") {
      JSON_ENFORCE_OBJECT();
      return std::make_shared<pipeline::TextEncoder>(item, profileStat, logger);
    } else if (item.key() == "text-generator") {
      JSON_ENFORCE_OBJECT();
      return std::make_shared<pipeline::TextGenerator>(item, profileStat, logger);
    } else if (item.key() == "image-encoder") {
      return std::make_shared<pipeline::ImageEncoder>(item, profileStat, logger);
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown config key: " + item.key());
    }
  }
  return nullptr;
}

pipeline::Node::Node(qualla::json config) {
  m_config = config;
  m_name   = "node" + std::to_string(s_nameCounter.fetch_add(1u));
}

GenieNode_Handle_t pipeline::Node::add(std::shared_ptr<pipeline::Node> node) {
  return (GenieNode_Handle_t)s_manager.add(node);
}

std::shared_ptr<pipeline::Node> pipeline::Node::get(GenieNode_Handle_t handle) {
  return s_manager.get((qnn::util::Handle_t)handle);
}

void pipeline::Node::remove(GenieNode_Handle_t handle) {
  s_manager.remove((qnn::util::Handle_t)handle);
}

int32_t pipeline::Node::bindPipeline(Pipeline& pipeline) {
  if (m_pipeline) {
    throw Exception(GENIE_STATUS_ERROR_GENERAL, "Node already bound to Pipeline");
  }
  m_pipeline = &pipeline;
  return GENIE_STATUS_SUCCESS;
}

std::string pipeline::Node::getName() const { return m_name; }

int32_t pipeline::Node::execute(void* userData) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "execute not supported on Node");
}

int32_t pipeline::Node::save(const std::string& name) { return GENIE_STATUS_SUCCESS; }

int32_t pipeline::Node::restore(const std::string& name) { return GENIE_STATUS_SUCCESS; }

int32_t pipeline::Node::setPriority(const std::string engine,
                                    const GeniePipeline_Priority_t priority) {
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::Node::setOemkey(const std::string& oemKey) { return GENIE_STATUS_SUCCESS; }

int32_t pipeline::Node::applyLora(std::string loraAdapterName, std::string engine) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "applyLora not supported in node");
}

int32_t pipeline::Node::applyLoraStrength(std::string tensorName, std::string engine, float alpha) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "applyLoraStrength not supported in node");
}

GenieEngine_Handle_t pipeline::Node::getEngineHandle(const std::string& engineRole) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "getEngineHandle not supported by Node");
  return nullptr;
}

int32_t pipeline::Node::bindEngine(const std::string& engineRole, std::shared_ptr<Engine> engine) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "bindEngine not supported by Node");
  return GENIE_STATUS_ERROR_GENERAL;
}

GenieSampler_Handle_t pipeline::Node::getSamplerHandle() {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "getSamplerHandle not supported by Node");
  return nullptr;
}

GenieTokenizer_Handle_t pipeline::Node::getTokenizerHandle() {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "getTokenizerHandle not supported by Node");
  return nullptr;
}

// Input Modality setters
int32_t pipeline::Node::setTextInputData(GenieNode_IOName_t nodeIOName, const char* txt) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "setTextInputData not supported in node");
}

int32_t pipeline::Node::setEmbeddingInputData(GenieNode_IOName_t nodeIOName,
                                              const void* embedding,
                                              size_t embeddingSize) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "setEmbeddingInputData not supported in node");
}

int32_t pipeline::Node::setImageInputData(GenieNode_IOName_t nodeIOName,
                                          const void* imageData,
                                          size_t imageSize) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "setImageInputData not supported in node");
}

// Output Modality setters
int32_t pipeline::Node::setTextOutputCallback(GenieNode_IOName_t nodeIOName,
                                              GenieNode_TextOutput_Callback_t callback) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "setTextOutputCallback not supported in node");
}

int32_t pipeline::Node::setEmbeddingOutputCallback(GenieNode_IOName_t nodeIOName,
                                                   GenieNode_EmbeddingOutputCallback_t callback) {
  throw Exception(GENIE_STATUS_ERROR_GENERAL, "setEmbeddingOutputCallback not supported in node");
}

void pipeline::Node::bindLogger(std::unordered_set<std::shared_ptr<genie::log::Logger>>& logger) {
  for (auto it : logger) {
    it->incrementUseCount();
    m_logger.insert(it);
  }
}

void pipeline::Node::unbindLogger() {
  for (auto it : m_logger) it->decrementUseCount();
  m_logger.clear();
}

std::unordered_set<std::shared_ptr<genie::log::Logger>>& pipeline::Node::getLogger() {
  return m_logger;
}

void pipeline::Node::bindProfiler(std::unordered_set<std::shared_ptr<Profiler>>& profiler) {
  for (auto it : profiler) {
    it->incrementUseCount();
    m_profiler.insert(it);
  }
}

void pipeline::Node::unbindProfiler() {
  for (auto it : m_profiler) it->decrementUseCount();
  m_profiler.clear();
}

std::unordered_set<std::shared_ptr<Profiler>>& pipeline::Node::getProfiler() { return m_profiler; }

pipeline::Node::~Node() {}