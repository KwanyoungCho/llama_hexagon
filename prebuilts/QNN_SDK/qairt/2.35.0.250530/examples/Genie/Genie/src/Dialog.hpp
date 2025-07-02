//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <atomic>
#include <functional>
#include <memory>

#include "Engine.hpp"
#include "GenieDialog.h"
#include "GenieEngine.h"
#include "GenieNode.h"
#include "LogUtils.hpp"
#include "Logger.hpp"
#include "Profile.hpp"
#include "Sampler.hpp"
#include "Tokenizer.hpp"
#include "Util/HandleManager.hpp"
#include "qualla/DialogCallback.hpp"
#include "qualla/dialog.hpp"

namespace genie {

class Dialog {
 public:
  class Config {
   public:
    static GenieDialogConfig_Handle_t add(std::shared_ptr<Config> config);
    static std::shared_ptr<Config> get(GenieDialogConfig_Handle_t handle);
    static void remove(GenieDialogConfig_Handle_t handle);
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

  static void validateDialogConfig(const qualla::json& config);
  static void translateDialogConfig(const qualla::json& genieConfig, qualla::json& quallaConfig);
  static GenieDialog_Handle_t add(std::shared_ptr<Dialog> dialog);
  static std::shared_ptr<Dialog> get(GenieDialog_Handle_t handle);
  static void remove(GenieDialog_Handle_t handle);
  static GenieSampler_Handle_t getSamplerHandle(std::shared_ptr<genie::Dialog> dialog);
  static GenieTokenizer_Handle_t getTokenizerHandle(std::shared_ptr<genie::Dialog> dialog);

  qualla::DialogCallback dialogCallback;

  void initDialog(qualla::json& config,
                  std::shared_ptr<ProfileStat> profileStat,
                  std::shared_ptr<genie::log::Logger> logger = nullptr);

  Dialog(qualla::json& config,
         std::shared_ptr<ProfileStat> profileStat,
         std::shared_ptr<genie::log::Logger> logger = nullptr);
  Dialog(std::shared_ptr<Config> config,
         std::shared_ptr<ProfileStat> profileStat,
         std::shared_ptr<genie::log::Logger> logger = nullptr);
  ~Dialog();

  std::string getName();

  Dialog(const Dialog&)            = delete;
  Dialog& operator=(const Dialog&) = delete;
  Dialog(Dialog&&)                 = delete;
  Dialog& operator=(Dialog&&)      = delete;

  int32_t query(const char* queryStr,
                GenieDialog_SentenceCode_t sentenceCode,
                GenieDialog_QueryCallback_t callback,
                const void* userData,
                std::shared_ptr<ProfileStat> profileStat);

  int32_t query(const char* queryStr,
                GenieNode_TextOutput_SentenceCode_t sentenceCode,
                GenieNode_TextOutput_Callback_t callback,
                const void* userData,
                std::shared_ptr<ProfileStat> profileStat);

  int32_t embeddingQuery(const void* embeddings,
                         const uint32_t embeddingsSize,
                         GenieDialog_SentenceCode_t sentenceCode,
                         GenieDialog_TokenToEmbeddingCallback_t t2eCallback,
                         GenieDialog_QueryCallback_t callback,
                         const void* userData,
                         std::shared_ptr<ProfileStat> profileStat);

  int32_t embeddingQuery(const void* embeddings,
                         const uint32_t embeddingsSize,
                         GenieNode_TextOutput_SentenceCode_t sentenceCode,
                         GenieNode_TextOutput_Callback_t callback,
                         const void* userData,
                         std::shared_ptr<ProfileStat> profileStat);

  int32_t tokenQuery(const uint32_t* tokens,
                     const uint32_t sizeInputTokens,
                     GenieDialog_SentenceCode_t sentenceCode,
                     GenieDialog_TokenQueryCallback_t callback,
                     const void* userData,
                     std::shared_ptr<ProfileStat> profileStat);

  int32_t save(const std::string&);
  int32_t restore(const std::string&);
  void reset();

  int32_t signalAction(GenieDialog_Action_t action);

  void setStopSequence(const char* newStopSeqs);

  int32_t applyLora(std::string loraAdapterName, std::string engine);
  int32_t applyLoraStrength(std::string tensorName, std::string engine, float alpha);
  int32_t setPriority(std::string engine, const GenieDialog_Priority_t priority);
  int32_t setOemkey(const std::string& oemKey);

  void bindProfiler(std::unordered_set<std::shared_ptr<Profiler>>& profiler);
  void unbindProfiler();
  std::unordered_set<std::shared_ptr<Profiler>>& getProfiler();

  void bindLogger(std::unordered_set<std::shared_ptr<genie::log::Logger>>& log);
  void unbindLogger();
  std::unordered_set<std::shared_ptr<genie::log::Logger>>& getLogger();

  GenieEngine_Handle_t getEngineHandle(const std::string& engineRole);
  int32_t bindEngine(const std::string& engineRole, std::shared_ptr<Engine> engine);

  int32_t getInputQuantParam(std::string& dataType,
                             double& scale,
                             int32_t& offset,
                             size_t& byteWidth);

 protected:
  // Protected visibility for genie-ppl-run
  std::unique_ptr<qualla::Dialog> m_quallaDialog;

 private:
  uint32_t m_tokenLimit{UINT32_MAX};
  std::atomic<bool> m_abort{false};
  std::atomic<uint32_t> m_activeQuery{0};
  static qnn::util::HandleManager<Dialog> s_manager;
  static std::atomic<std::uint32_t> s_nameCounter;
  std::string m_name;
  GenieSampler_Handle_t m_samplerHandle;
  GenieTokenizer_Handle_t m_tokenizerHandle;
  std::unordered_set<std::shared_ptr<Profiler>> m_profiler;
  std::unordered_set<std::shared_ptr<genie::log::Logger>> m_logger;
};
}  // namespace genie
