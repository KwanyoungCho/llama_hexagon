//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include "Exception.hpp"
#include "GenieNode.h"
#include "Macro.hpp"
#include "Util/HandleManager.hpp"
#include "pipeline/Node.hpp"
#include "qualla/detail/json.hpp"
using namespace genie;

#ifdef __cplusplus
extern "C" {
#endif

GENIE_API
Genie_Status_t GenieNodeConfig_createFromJson(const char* str,
                                              GenieNodeConfig_Handle_t* configHandle) {
  try {
    GENIE_ENSURE(str, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    auto config = std::make_shared<pipeline::Node::Config>(str);
    GENIE_ENSURE(config, GENIE_STATUS_ERROR_MEM_ALLOC);
    *configHandle = genie::pipeline::Node::Config::add(config);
  } catch (const qualla::json::parse_error& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_JSON_FORMAT;
  } catch (const Exception& e) {
    std::cerr << e.what() << std::endl;
    return e.status();
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieNodeConfig_bindProfiler(const GenieNodeConfig_Handle_t configHandle,
                                            const GenieProfile_Handle_t profileHandle) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(profileHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto configObj = genie::pipeline::Node::Config::get(configHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto profiler = Profiler::get(profileHandle);
    GENIE_ENSURE(profiler, GENIE_STATUS_ERROR_INVALID_HANDLE);
    configObj->bindProfiler(profiler);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieNodeConfig_bindLogger(const GenieNodeConfig_Handle_t configHandle,
                                          const GenieLog_Handle_t logHandle) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(logHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto configObj = genie::pipeline::Node::Config::get(configHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto logger = genie::log::Logger::getLogger(logHandle);
    GENIE_ENSURE(logger, GENIE_STATUS_ERROR_INVALID_HANDLE);
    configObj->bindLogger(logger);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieNodeConfig_free(const GenieNodeConfig_Handle_t configHandle) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    // Check if the pipeline actually exists
    auto configObj = genie::pipeline::Node::Config::get(configHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    configObj->unbindProfiler();
    configObj->unbindLogger();
    genie::pipeline::Node::Config::remove(configHandle);
  } catch (const std::exception& e) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieNode_create(const GenieNodeConfig_Handle_t nodeConfigHandle,
                                GenieNode_Handle_t* nodeHandle) {
  try {
    const uint64_t startTime = genie::getTimeStampInUs();
    std::shared_ptr<ProfileStat> profileStat;
    GENIE_ENSURE(nodeConfigHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);

    // Get config object
    auto configObj = genie::pipeline::Node::Config::get(nodeConfigHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    if (!configObj->getProfiler().empty()) {
      profileStat = std::make_shared<ProfileStat>(
          GENIE_PROFILE_EVENTTYPE_NODE_CREATE, startTime, "", GENIE_PROFILE_COMPONENTTYPE_NODE);
    }

    std::shared_ptr<genie::log::Logger> logger;
    if (!configObj->getLogger().empty())
      logger = *configObj->getLogger().begin();
    else
      logger = nullptr;

    // Create node
    auto node = genie::pipeline::Node::createNode(configObj, profileStat, logger);
    GENIE_ENSURE(node, GENIE_STATUS_ERROR_MEM_ALLOC);

    // Create Handle
    *nodeHandle = genie::pipeline::Node::add(node);
    node->bindProfiler(configObj->getProfiler());

    const uint64_t stopTime = genie::getTimeStampInUs();
    if (profileStat) {
      profileStat->setComponentId(node->getName().c_str());
      profileStat->setDuration(stopTime - startTime);
    }
    const std::unordered_set<std::shared_ptr<Profiler>> profiler = node->getProfiler();
    for (auto it : profiler) it->addProfileStat(profileStat);

  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  // Return SUCCESS
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieNode_free(const GenieNode_Handle_t nodeHandle) {
  try {
    GENIE_ENSURE(nodeHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    std::shared_ptr<ProfileStat> profileStat;
    std::unordered_set<std::shared_ptr<Profiler>> profiler;
    {
      // Check if the node actually exists
      auto node = genie::pipeline::Node::get(nodeHandle);
      GENIE_ENSURE(node, GENIE_STATUS_ERROR_INVALID_HANDLE);
    }
    genie::pipeline::Node::remove(nodeHandle);

  } catch (const std::exception& e) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieNode_setData(const GenieNode_Handle_t nodeHandle,
                                 const GenieNode_IOName_t nodeIOName,
                                 const void* data,
                                 const size_t dataSize,
                                 const char* dataConfig) {
  int32_t status;
  (void)dataConfig;
  try {
    GENIE_ENSURE(nodeHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto node = genie::pipeline::Node::get(nodeHandle);
    GENIE_ENSURE(node, GENIE_STATUS_ERROR_INVALID_HANDLE);
    if (nodeIOName == GENIE_NODE_IMAGE_ENCODER_IMAGE_INPUT) {
      status = node->setImageInputData(nodeIOName, data, dataSize);
    } else if (nodeIOName == GENIE_NODE_TEXT_ENCODER_TEXT_INPUT) {
      status = node->setTextInputData(nodeIOName, (char*)data);
    } else if (nodeIOName == GENIE_NODE_TEXT_GENERATOR_EMBEDDING_INPUT) {
      status = node->setEmbeddingInputData(nodeIOName, data, dataSize);
    } else {
      std::cerr << "setData not supported on node input " << nodeIOName << std::endl;
      return GENIE_STATUS_ERROR_GENERAL;
    }
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }

  return status;
}

GENIE_API
Genie_Status_t GenieNode_setTextCallback(const GenieNode_Handle_t nodeHandle,
                                         const GenieNode_IOName_t nodeIOName,
                                         const GenieNode_TextOutput_Callback_t callback) {
  int32_t status;

  try {
    GENIE_ENSURE(nodeHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto node = genie::pipeline::Node::get(nodeHandle);
    GENIE_ENSURE(node, GENIE_STATUS_ERROR_INVALID_HANDLE);
    status = node->setTextOutputCallback(nodeIOName, callback);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }

  return status;
}

GENIE_API
Genie_Status_t GenieNode_setEmbeddingCallback(const GenieNode_Handle_t nodeHandle,
                                              const GenieNode_IOName_t nodeIOName,
                                              const GenieNode_EmbeddingOutputCallback_t callback) {
  int32_t status;

  try {
    GENIE_ENSURE(nodeHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto node = genie::pipeline::Node::get(nodeHandle);
    GENIE_ENSURE(node, GENIE_STATUS_ERROR_INVALID_HANDLE);
    status = node->setEmbeddingOutputCallback(nodeIOName, callback);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }

  return status;
}

GENIE_API
Genie_Status_t GenieNode_applyLora(const GenieNode_Handle_t nodeHandle,
                                   const char* engine,
                                   const char* loraAdapterName) {
  int32_t status;
  try {
    GENIE_ENSURE(nodeHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto node = genie::pipeline::Node::get(nodeHandle);
    GENIE_ENSURE(node, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(engine, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    std::string eng(engine);
    GENIE_ENSURE(loraAdapterName, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    std::string loraName(loraAdapterName);
    status = node->applyLora(loraName, eng);
  } catch (const std::exception& e) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return status;
}

GENIE_API
Genie_Status_t GenieNode_setLoraStrength(const GenieNode_Handle_t nodeHandle,
                                         const char* engine,
                                         const char* tensorName,
                                         const float alpha) {
  int32_t status;
  try {
    GENIE_ENSURE(nodeHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto node = genie::pipeline::Node::get(nodeHandle);
    GENIE_ENSURE(node, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(engine, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    std::string eng(engine);
    GENIE_ENSURE(tensorName, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    std::string alphaTensorName(tensorName);
    GENIE_ENSURE_NOT_EMPTY(alphaTensorName, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    status = node->applyLoraStrength(tensorName, eng, alpha);
  } catch (const std::exception& e) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return status;
}

GENIE_API
Genie_Status_t GenieNode_getSampler(const GenieNode_Handle_t nodeHandle,
                                    GenieSampler_Handle_t* nodeSamplerHandle) {
  try {
    GENIE_ENSURE(nodeHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto node = genie::pipeline::Node::get(nodeHandle);
    GENIE_ENSURE(node, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(nodeSamplerHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    *nodeSamplerHandle = node->getSamplerHandle();
    GENIE_ENSURE(*nodeSamplerHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GET_HANDLE_FAILED;
  }

  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieNode_getTokenizer(const GenieNode_Handle_t nodeHandle,
                                      GenieTokenizer_Handle_t* tokenizerHandle) {
  try {
    GENIE_ENSURE(nodeHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto node = genie::pipeline::Node::get(nodeHandle);
    GENIE_ENSURE(node, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(tokenizerHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    *tokenizerHandle = node->getTokenizerHandle();
    GENIE_ENSURE(*tokenizerHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GET_HANDLE_FAILED;
  }

  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieNode_getEngine(const GenieNode_Handle_t nodeHandle,
                                   const char* engineRole,
                                   GenieEngine_Handle_t* nodeEngineHandle) {
  try {
    GENIE_ENSURE(nodeHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto node = genie::pipeline::Node::get(nodeHandle);
    GENIE_ENSURE(node, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(nodeEngineHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(engineRole, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    *nodeEngineHandle = node->getEngineHandle(engineRole);
    GENIE_ENSURE(*nodeEngineHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GET_HANDLE_FAILED;
  }

  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieNode_bindEngine(const GenieNode_Handle_t nodeHandle,
                                    const char* engineRole,
                                    const GenieEngine_Handle_t engineHandle) {
  try {
    GENIE_ENSURE(nodeHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto node = genie::pipeline::Node::get(nodeHandle);
    GENIE_ENSURE(node, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(engineHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto engine = genie::Engine::get(engineHandle);
    GENIE_ENSURE(engine, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(engineRole, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    node->bindEngine(engineRole, engine);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif
