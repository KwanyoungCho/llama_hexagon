//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <memory>

#include "Exception.hpp"
#include "Pipeline.hpp"

using namespace genie;

//=============================================================================
// Pipeline::Config functions
//=============================================================================

qnn::util::HandleManager<pipeline::Pipeline::Config> pipeline::Pipeline::Config::s_manager;

GeniePipelineConfig_Handle_t pipeline::Pipeline::Config::add(
    std::shared_ptr<pipeline::Pipeline::Config> config) {
  return (GeniePipelineConfig_Handle_t)s_manager.add(config);
}

std::shared_ptr<pipeline::Pipeline::Config> pipeline::Pipeline::Config::get(
    GeniePipelineConfig_Handle_t handle) {
  return s_manager.get((qnn::util::Handle_t)handle);
}

void pipeline::Pipeline::Config::remove(GeniePipelineConfig_Handle_t handle) {
  s_manager.remove((qnn::util::Handle_t)handle);
}

pipeline::Pipeline::Config::Config(const char* configStr) {
  if (configStr != nullptr && (configStr[0] != '\0')) {
    m_config = qualla::json::parse(configStr);
  }
}

qualla::json& pipeline::Pipeline::Config::getJson() { return m_config; }

void pipeline::Pipeline::Config::bindLogger(std::shared_ptr<genie::log::Logger> logger) {
  if (!logger) return;
  logger->incrementUseCount();
  m_logger.insert(logger);
}

void pipeline::Pipeline::Config::unbindLogger() {
  for (auto it : m_logger) it->decrementUseCount();
  m_logger.clear();
}

std::unordered_set<std::shared_ptr<genie::log::Logger>>& pipeline::Pipeline::Config::getLogger() {
  return m_logger;
}

void pipeline::Pipeline::Config::bindProfiler(std::shared_ptr<Profiler> profiler) {
  if (!profiler) return;
  profiler->incrementUseCount();
  m_profiler.insert(profiler);
}

void pipeline::Pipeline::Config::unbindProfiler() {
  for (auto it : m_profiler) it->decrementUseCount();
  m_profiler.clear();
}

std::unordered_set<std::shared_ptr<Profiler>>& pipeline::Pipeline::Config::getProfiler() {
  return m_profiler;
}

//=============================================================================
// Pipeline functions
//=============================================================================

qnn::util::HandleManager<pipeline::Pipeline> pipeline::Pipeline::s_manager;
std::atomic<std::uint32_t> pipeline::Pipeline::s_nameCounter{0u};

pipeline::Pipeline::Pipeline(std::shared_ptr<Config> config,
                             std::shared_ptr<ProfileStat> profileStat,
                             std::shared_ptr<genie::log::Logger> logger) {
  auto env = qualla::Env::create(qualla::json{});
  if (logger) env->bindLogger(logger);
  qualla::json quallaConfig;
  m_name = "pipeline" + std::to_string(s_nameCounter.fetch_add(1u));
}

void pipeline::Pipeline::setupAccumultator(size_t accumulatorSize) {
  // Create Accumulator for each input to generator
  m_accumulator = std::make_shared<Accumulator>(accumulatorSize);
}

GeniePipeline_Handle_t pipeline::Pipeline::add(std::shared_ptr<pipeline::Pipeline> pipeline) {
  return (GeniePipeline_Handle_t)s_manager.add(pipeline);
}

std::shared_ptr<pipeline::Pipeline> pipeline::Pipeline::get(GeniePipeline_Handle_t handle) {
  return s_manager.get((qnn::util::Handle_t)handle);
}

void pipeline::Pipeline::remove(GeniePipeline_Handle_t handle) {
  s_manager.remove((qnn::util::Handle_t)handle);
}

int32_t pipeline::Pipeline::addNode(std::shared_ptr<Node> node) {
  m_nodes.insert(node);
  node->bindPipeline(*this);
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::Pipeline::pipelineExecute(void* userData) {
  for (auto node : m_nodes) {
    node->execute(userData);
  }
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::Pipeline::save(const std::string& name) { /* TODO */
  for (auto node : m_nodes) {
    return node->save(name);
  }
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::Pipeline::restore(const std::string& name) { /* TODO */
  for (auto node : m_nodes) {
    return node->restore(name);
  }
  return GENIE_STATUS_SUCCESS;
}

void pipeline::Pipeline::reset() { /* TODO */
}

int32_t pipeline::Pipeline::setPriority(const std::string engine,
                                        const GeniePipeline_Priority_t priority) {
  for (auto node : m_nodes) {
    return node->setPriority(engine, priority);
  }
  return GENIE_STATUS_SUCCESS;
}

int32_t pipeline::Pipeline::setOemkey(const std::string& oemKey) { /* TODO */
  for (auto node : m_nodes) {
    return node->setOemkey(oemKey);
  }
  return GENIE_STATUS_SUCCESS;
}

pipeline::Pipeline::~Pipeline() {}

std::string pipeline::Pipeline::getName() const { return m_name; }

void pipeline::Pipeline::bindLogger(
    std::unordered_set<std::shared_ptr<genie::log::Logger>>& logger) {
  for (auto it : logger) {
    it->incrementUseCount();
    m_logger.insert(it);
  }
}

void pipeline::Pipeline::unbindLogger() {
  for (auto it : m_logger) it->decrementUseCount();
  m_logger.clear();
}

std::unordered_set<std::shared_ptr<genie::log::Logger>>& pipeline::Pipeline::getLogger() {
  return m_logger;
}

void pipeline::Pipeline::bindProfiler(std::unordered_set<std::shared_ptr<Profiler>>& profiler) {
  for (auto it : profiler) {
    it->incrementUseCount();
    m_profiler.insert(it);
  }
}

void pipeline::Pipeline::unbindProfiler() {
  for (auto it : m_profiler) it->decrementUseCount();
  m_profiler.clear();
}

std::unordered_set<std::shared_ptr<Profiler>>& pipeline::Pipeline::getProfiler() {
  return m_profiler;
}
