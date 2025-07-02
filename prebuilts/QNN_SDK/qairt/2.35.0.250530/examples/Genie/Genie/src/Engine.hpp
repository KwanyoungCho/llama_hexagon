//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once
#include <memory>

#include "GenieEngine.h"
#include "Util/HandleManager.hpp"
#include "qualla/engine.hpp"
#include "qualla/env.hpp"

enum LORA_VERSION : uint8_t {
  GENIE_LORA_VERSION_V1        = 0x01,
  GENIE_LORA_VERSION_V2        = 0x02,
  GENIE_LORA_VERSION_V3        = 0x03,
  GENIE_LORA_VERSION_UNDEFINED = 0xFF
};

namespace genie {
class Engine {
 public:
  class EngineConfig {
   public:
    static GenieEngineConfig_Handle_t add(std::shared_ptr<EngineConfig> config);

    static std::shared_ptr<EngineConfig> get(GenieEngineConfig_Handle_t handle);

    static void remove(GenieEngineConfig_Handle_t handle);

    EngineConfig(const char* configStr);

    qualla::json& getJson();

   private:
    static qnn::util::HandleManager<EngineConfig> s_manager;
    qualla::json m_config;
  };

  static GenieEngine_Handle_t add(std::shared_ptr<Engine> engine);
  static std::shared_ptr<Engine> get(GenieEngine_Handle_t handle);
  static void remove(GenieEngine_Handle_t handle);
  Engine(std::shared_ptr<EngineConfig>& config);
  Engine(std::shared_ptr<qualla::Engine> quallaEngine, const std::string& name);
  ~Engine();

  bool checkIsEngineBound();
  Engine(const Engine&)            = delete;
  Engine& operator=(const Engine&) = delete;
  Engine(Engine&&)                 = delete;
  Engine& operator=(Engine&&)      = delete;

  std::shared_ptr<qualla::Engine> getEngine() { return m_quallaEngine; }

 private:
  std::string m_name;
  std::shared_ptr<qualla::Engine> m_quallaEngine;
  std::shared_ptr<qualla::Env> m_env;
  std::unique_ptr<qualla::Context> m_context;
  static qnn::util::HandleManager<Engine> s_manager;
  static std::atomic<std::uint32_t> s_nameCounter;
};

}  // namespace genie
