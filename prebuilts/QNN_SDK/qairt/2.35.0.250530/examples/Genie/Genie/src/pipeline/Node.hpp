//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once
#ifndef NODE_HPP
#define NODE_HPP
#include <memory>

#include "Accumulator.hpp"
#include "Dialog.hpp"
#include "Embedding.hpp"
#include "GenieNode.h"
#include "GeniePipeline.h"

namespace genie {
namespace pipeline {

class Pipeline;
class Node {
 public:
  class Config {
   public:
    static GenieNodeConfig_Handle_t add(std::shared_ptr<Config> config);
    static std::shared_ptr<Config> get(GenieNodeConfig_Handle_t handle);
    static void remove(GenieNodeConfig_Handle_t handle);
    Config(const char* configStr);
    qualla::json& getJson() { return m_config; };
    void bindProfiler(std::shared_ptr<Profiler> profiler);
    void unbindProfiler();
    std::unordered_set<std::shared_ptr<Profiler>>& getProfiler();
    void bindLogger(std::shared_ptr<genie::log::Logger> log);
    void unbindLogger();
    std::unordered_set<std::shared_ptr<genie::log::Logger>>& getLogger();

   private:
    static qnn::util::HandleManager<Config> s_manager;
    qualla::json m_config;
    std::unordered_set<std::shared_ptr<Profiler>> m_profiler;
    std::unordered_set<std::shared_ptr<genie::log::Logger>> m_logger;
  };

  static std::shared_ptr<Node> createNode(std::shared_ptr<Config> config,
                                          std::shared_ptr<ProfileStat> profileStat,
                                          std::shared_ptr<genie::log::Logger> logger = nullptr);
  Node(qualla::json config);
  // Node Utilities
  static GenieNode_Handle_t add(std::shared_ptr<Node> node);
  static std::shared_ptr<Node> get(GenieNode_Handle_t handle);

  void bindProfiler(std::unordered_set<std::shared_ptr<Profiler>>& profiler);
  void unbindProfiler();
  std::unordered_set<std::shared_ptr<Profiler>>& getProfiler();

  void bindLogger(std::unordered_set<std::shared_ptr<genie::log::Logger>>& log);
  void unbindLogger();
  std::unordered_set<std::shared_ptr<genie::log::Logger>>& getLogger();
  std::string getName() const;

  static void remove(GenieNode_Handle_t handle);
  virtual int32_t bindPipeline(Pipeline& pipeline);
  virtual int32_t execute(void* userData) = 0;

  bool isTypeGenerator() { return m_typeGenerator; };
  bool isConnected() { return m_isConnected; };
  virtual int32_t save(const std::string&);
  virtual int32_t restore(const std::string&);
  virtual void reset(){};
  virtual int32_t setPriority(std::string engine, const GeniePipeline_Priority_t priority);
  virtual int32_t setOemkey(const std::string& oemKey);

  virtual int32_t applyLora(std::string loraAdapterName, std::string engine);
  virtual int32_t applyLoraStrength(std::string tensorName, std::string engine, float alpha);

  virtual GenieEngine_Handle_t getEngineHandle(const std::string& engineRole);
  virtual int32_t bindEngine(const std::string& engineRole, std::shared_ptr<Engine> engine);
  virtual GenieSampler_Handle_t getSamplerHandle();
  virtual GenieTokenizer_Handle_t getTokenizerHandle();

  // Input Modality setters
  virtual int32_t setTextInputData(GenieNode_IOName_t nodeIOName, const char* txt);
  virtual int32_t setEmbeddingInputData(GenieNode_IOName_t nodeIOName,
                                        const void* embedding,
                                        size_t embeddingSize);
  virtual int32_t setImageInputData(GenieNode_IOName_t nodeIOName,
                                    const void* imageData,
                                    size_t imageSize);

  // Output Modality setters
  virtual int32_t setTextOutputCallback(GenieNode_IOName_t nodeIOName,
                                        GenieNode_TextOutput_Callback_t callback);
  virtual int32_t setEmbeddingOutputCallback(GenieNode_IOName_t nodeIOName,
                                             GenieNode_EmbeddingOutputCallback_t callback);
  void markConnected() { m_isConnected = true; };
  virtual ~Node();

 protected:
  bool m_typeGenerator = false;
  Pipeline* m_pipeline = nullptr;
  qualla::json m_config;

 private:
  static qnn::util::HandleManager<Node> s_manager;
  std::unordered_set<std::shared_ptr<Profiler>> m_profiler;
  std::unordered_set<std::shared_ptr<genie::log::Logger>> m_logger;
  static std::atomic<std::uint32_t> s_nameCounter;
  std::string m_name;
  bool m_isConnected = false;
};
}  // namespace pipeline
}  // namespace genie
#endif  // NODE_HPP