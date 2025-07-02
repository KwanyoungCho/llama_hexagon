//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include "Exception.hpp"
#include "GenieNode.h"
#include "GeniePipeline.h"
#include "Macro.hpp"
#include "Util/HandleManager.hpp"
#include "pipeline/Node.hpp"
#include "pipeline/Pipeline.hpp"
#include "qualla/detail/json.hpp"

using namespace genie;

#ifdef __cplusplus
extern "C" {
#endif

GENIE_API
Genie_Status_t GeniePipelineConfig_createFromJson(const char* str,
                                                  GeniePipelineConfig_Handle_t* configHandle) {
  try {
    GENIE_ENSURE(str, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    auto config = std::make_shared<pipeline::Pipeline::Config>(str);
    GENIE_ENSURE(config, GENIE_STATUS_ERROR_MEM_ALLOC);
    *configHandle = genie::pipeline::Pipeline::Config::add(config);
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
Genie_Status_t GeniePipelineConfig_bindProfiler(const GeniePipelineConfig_Handle_t configHandle,
                                                const GenieProfile_Handle_t profileHandle) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(profileHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto configObj = genie::pipeline::Pipeline::Config::get(configHandle);
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
Genie_Status_t GeniePipelineConfig_bindLogger(const GeniePipelineConfig_Handle_t configHandle,
                                              const GenieLog_Handle_t logHandle) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(logHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto configObj = genie::pipeline::Pipeline::Config::get(configHandle);
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
Genie_Status_t GeniePipelineConfig_free(const GeniePipelineConfig_Handle_t configHandle) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    // Check if the pipeline actually exists
    auto configObj = genie::pipeline::Pipeline::Config::get(configHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    configObj->unbindProfiler();
    configObj->unbindLogger();
    genie::pipeline::Pipeline::Config::remove(configHandle);
  } catch (const std::exception& e) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GeniePipeline_create(const GeniePipelineConfig_Handle_t configHandle,
                                    GeniePipeline_Handle_t* pipelineHandle) {
  try {
    const uint64_t startTime = genie::getTimeStampInUs();
    std::shared_ptr<ProfileStat> profileStat;
    GENIE_ENSURE(pipelineHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);

    // Get config object
    auto configObj = genie::pipeline::Pipeline::Config::get(configHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    if (!configObj->getProfiler().empty()) {
      profileStat = std::make_shared<ProfileStat>(GENIE_PROFILE_EVENTTYPE_PIPELINE_CREATE,
                                                  startTime,
                                                  "",
                                                  GENIE_PROFILE_COMPONENTTYPE_PIPELINE);
    }

    std::shared_ptr<genie::log::Logger> logger;
    if (!configObj->getLogger().empty())
      logger = *configObj->getLogger().begin();  // currently uses one logger with one pipeline
    else
      logger = nullptr;
    // Create pipeline
    auto pipeline = std::make_shared<genie::pipeline::Pipeline>(configObj, profileStat, logger);
    GENIE_ENSURE(pipeline, GENIE_STATUS_ERROR_MEM_ALLOC);

    // Create Handle
    *pipelineHandle = genie::pipeline::Pipeline::add(pipeline);

    pipeline->bindProfiler(configObj->getProfiler());

    const uint64_t stopTime = genie::getTimeStampInUs();
    if (profileStat) {
      profileStat->setComponentId(pipeline->getName().c_str());
      profileStat->setDuration(stopTime - startTime);
    }
    const std::unordered_set<std::shared_ptr<Profiler>> profiler = pipeline->getProfiler();
    for (auto it : profiler) it->addProfileStat(profileStat);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  // Return SUCCESS
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GeniePipeline_save(const GeniePipeline_Handle_t pipelineHandle, const char* path) {
  int32_t status;

  try {
    GENIE_ENSURE(pipelineHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto pipeline = genie::pipeline::Pipeline::get(pipelineHandle);
    GENIE_ENSURE(pipeline, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(path, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    status = pipeline->save(path);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }

  return status;
}

GENIE_API
Genie_Status_t GeniePipeline_restore(const GeniePipeline_Handle_t pipelineHandle,
                                     const char* path) {
  int32_t status;

  try {
    GENIE_ENSURE(pipelineHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto pipeline = genie::pipeline::Pipeline::get(pipelineHandle);
    GENIE_ENSURE(pipeline, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(path, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    status = pipeline->restore(path);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }

  return status;
}

GENIE_API
Genie_Status_t GeniePipeline_reset(const GeniePipeline_Handle_t pipelineHandle) {
  try {
    GENIE_ENSURE(pipelineHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto pipeline = genie::pipeline::Pipeline::get(pipelineHandle);
    GENIE_ENSURE(pipeline, GENIE_STATUS_ERROR_INVALID_HANDLE);
    pipeline->reset();
  } catch (const std::exception& e) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GeniePipeline_setPriority(const GeniePipeline_Handle_t pipelineHandle,
                                         const char* engineRole,
                                         const GeniePipeline_Priority_t priority) {
  int32_t status;
  try {
    switch (priority) {
      case GENIE_PIPELINE_PRIORITY_LOW:
      case GENIE_PIPELINE_PRIORITY_NORMAL:
      case GENIE_PIPELINE_PRIORITY_NORMAL_HIGH:
      case GENIE_PIPELINE_PRIORITY_HIGH:
        // Do nothing
        break;
      default:
        return GENIE_STATUS_ERROR_INVALID_ARGUMENT;
    }
    GENIE_ENSURE(pipelineHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto pipeline = genie::pipeline::Pipeline::get(pipelineHandle);
    GENIE_ENSURE(pipeline, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(engineRole, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    status = pipeline->setPriority(engineRole, priority);
  } catch (const std::exception& e) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return status;
}

GENIE_API
Genie_Status_t GeniePipeline_setOemKey(const GeniePipeline_Handle_t pipelineHandle,
                                       const char* oemKey) {
  int32_t status;
  try {
    GENIE_ENSURE(pipelineHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto pipeline = genie::pipeline::Pipeline::get(pipelineHandle);
    GENIE_ENSURE(pipeline, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(oemKey, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    status = pipeline->setOemkey(oemKey);
  } catch (const std::exception& e) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return status;
}

GENIE_API
Genie_Status_t GeniePipeline_addNode(const GeniePipeline_Handle_t pipelineHandle,
                                     const GenieNode_Handle_t nodeHandle) {
  int32_t status;
  try {
    GENIE_ENSURE(pipelineHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto pipeline = genie::pipeline::Pipeline::get(pipelineHandle);
    GENIE_ENSURE(pipeline, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(nodeHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto node = genie::pipeline::Node::get(nodeHandle);
    GENIE_ENSURE(node, GENIE_STATUS_ERROR_INVALID_HANDLE);
    status = pipeline->addNode(node);
  } catch (const std::exception& e) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return status;
}

GENIE_API
Genie_Status_t GeniePipeline_connect(const GeniePipeline_Handle_t pipelineHandle,
                                     const GenieNode_Handle_t producerHandle,
                                     const GenieNode_IOName_t producerName,
                                     const GenieNode_Handle_t consumerHandle,
                                     const GenieNode_IOName_t consumerName) {
  int32_t status;
  try {
    std::shared_ptr<genie::pipeline::Node> producer, consumer;
    if (producerHandle != nullptr) {
      producer = genie::pipeline::Node::get(producerHandle);
      GENIE_ENSURE(producer, GENIE_STATUS_ERROR_INVALID_HANDLE);
    }
    if (consumerHandle != nullptr) {
      consumer = genie::pipeline::Node::get(consumerHandle);
      GENIE_ENSURE(consumer, GENIE_STATUS_ERROR_INVALID_HANDLE);
      if (consumer->isTypeGenerator()) {
        // Enoders can only be connected to generators
        if (producer) {
          if (producer->isTypeGenerator()) {
            return GENIE_STATUS_ERROR_GENERAL;
          }
          producer->markConnected();
        }
      } else if (consumer == nullptr || producer == nullptr) {
        return GENIE_STATUS_ERROR_GENERAL;
      }
    }
    return GENIE_STATUS_SUCCESS;
  } catch (const std::exception& e) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
}

GENIE_API
Genie_Status_t GeniePipeline_execute(const GeniePipeline_Handle_t pipelineHandle, void* userData) {
  int32_t status;
  try {
    GENIE_ENSURE(pipelineHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto pipeline = genie::pipeline::Pipeline::get(pipelineHandle);
    GENIE_ENSURE(pipeline, GENIE_STATUS_ERROR_INVALID_HANDLE);
    status = pipeline->pipelineExecute(userData);
  } catch (const std::exception& e) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return status;
}

GENIE_API
Genie_Status_t GeniePipeline_free(const GeniePipeline_Handle_t pipelineHandle) {
  try {
    const uint64_t startTime = genie::getTimeStampInUs();
    GENIE_ENSURE(pipelineHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    std::shared_ptr<ProfileStat> profileStat;
    std::unordered_set<std::shared_ptr<Profiler>> profiler;
    {
      // Check if the pipeline actually exists
      auto pipeline = genie::pipeline::Pipeline::get(pipelineHandle);
      GENIE_ENSURE(pipeline, GENIE_STATUS_ERROR_INVALID_HANDLE);
      profiler = pipeline->getProfiler();
      if (!profiler.empty()) {
        profileStat = std::make_shared<ProfileStat>(GENIE_PROFILE_EVENTTYPE_PIPELINE_FREE,
                                                    startTime,
                                                    pipeline->getName(),
                                                    GENIE_PROFILE_COMPONENTTYPE_PIPELINE);
      }
      pipeline->unbindProfiler();
    }
    genie::pipeline::Pipeline::remove(pipelineHandle);

    const uint64_t stopTime = genie::getTimeStampInUs();
    if (profileStat) profileStat->setDuration(stopTime - startTime);
    for (auto it : profiler) it->addProfileStat(profileStat);
  } catch (const std::exception& e) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

#ifdef __cplusplus
}
#endif
