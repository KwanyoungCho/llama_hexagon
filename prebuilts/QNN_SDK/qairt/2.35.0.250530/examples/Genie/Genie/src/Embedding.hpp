//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <atomic>
#include <memory>

#include "Engine.hpp"
#include "GenieEmbedding.h"
#include "LogUtils.hpp"
#include "Logger.hpp"
#include "Profile.hpp"
#include "Util/HandleManager.hpp"
#include "qualla/encoder.hpp"

namespace genie {

class Embedding {
 public:
  class Config {
   public:
    static GenieEmbeddingConfig_Handle_t add(std::shared_ptr<Config> config);
    static std::shared_ptr<Config> get(GenieEmbeddingConfig_Handle_t handle);
    static void remove(GenieEmbeddingConfig_Handle_t handle);

    Config(const char* configStr);
    qualla::json& getJson();
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

  static GenieEmbedding_Handle_t add(std::shared_ptr<Embedding> embedding);
  static std::shared_ptr<Embedding> get(GenieEmbedding_Handle_t handle);
  static void remove(GenieEmbedding_Handle_t handle);

  static void validateEmbeddingConfig(const qualla::json& config, bool validateTextEncoder);
  static void translateEmbeddingConfig(const qualla::json& genieConfig, qualla::json& quallaConfig);

  void initEmbedding(qualla::json& config,
                     std::shared_ptr<ProfileStat> profileStat,
                     std::shared_ptr<genie::log::Logger> logger = nullptr);

  Embedding(qualla::json& config,
            std::shared_ptr<ProfileStat> profileStat,
            std::shared_ptr<genie::log::Logger> logger = nullptr);

  Embedding(std::shared_ptr<Config> config,
            std::shared_ptr<ProfileStat> profileStat,
            std::shared_ptr<genie::log::Logger> logger = nullptr);

  Embedding(const Embedding&)            = delete;
  Embedding& operator=(const Embedding&) = delete;
  Embedding(Embedding&&)                 = delete;
  Embedding& operator=(Embedding&&)      = delete;

  int32_t applyLora(std::string loraAdapterName, std::string engine);
  int32_t applyLoraStrength(std::string tensorName, std::string engine, float alpha);

  bool encode(const char* queryStr,
              std::vector<uint8_t>& outputEmbedding,
              std::shared_ptr<ProfileStat> profileStat);

  int32_t generate(const char* queryStr,
                   GenieEmbedding_GenerateCallback_t callback,
                   const void* userData,
                   std::shared_ptr<ProfileStat> profileStat);

  int32_t encode(const void* pixel_values,
                 const uint32_t inputSize,
                 std::vector<uint8_t>& outputEmbedding,
                 std::shared_ptr<ProfileStat> profileStat);

  void bindProfiler(std::unordered_set<std::shared_ptr<Profiler>>& profiler);
  void unbindProfiler();
  std::unordered_set<std::shared_ptr<Profiler>>& getProfiler();
  void bindLogger(std::unordered_set<std::shared_ptr<genie::log::Logger>>& log);
  void unbindLogger();
  std::unordered_set<std::shared_ptr<genie::log::Logger>>& getLogger();
  std::string getName();
  std::string getType();
  int32_t getOutputDimensions(std::vector<uint32_t>& dimensions);
  int32_t getOutputQuantParam(std::string& dataType,
                              double& scale,
                              int32_t& offset,
                              size_t& byteWidth);

 private:
  std::unique_ptr<qualla::Encoder> m_quallaEmbedding;
  static qnn::util::HandleManager<Embedding> s_manager;
  static std::atomic<std::uint32_t> s_nameCounter;
  std::string m_name;
  std::string m_type = "text";
  std::unordered_set<std::shared_ptr<Profiler>> m_profiler;
  std::unordered_set<std::shared_ptr<genie::log::Logger>> m_logger;
};
}  // namespace genie
