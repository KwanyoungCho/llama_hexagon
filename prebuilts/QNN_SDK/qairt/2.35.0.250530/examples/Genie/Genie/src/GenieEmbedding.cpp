//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include "Embedding.hpp"
#include "Exception.hpp"
#include "GenieEmbedding.h"
#include "Macro.hpp"
#include "Util/HandleManager.hpp"
#include "qualla/detail/json.hpp"

using namespace genie;

GENIE_API
Genie_Status_t GenieEmbeddingConfig_createFromJson(const char* str,
                                                   GenieEmbeddingConfig_Handle_t* configHandle) {
  try {
    GENIE_ENSURE(str, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    auto config = std::make_shared<Embedding::Config>(str);
    GENIE_ENSURE(config, GENIE_STATUS_ERROR_MEM_ALLOC);
    *configHandle = genie::Embedding::Config::add(config);
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
Genie_Status_t GenieEmbeddingConfig_bindProfiler(const GenieEmbeddingConfig_Handle_t configHandle,
                                                 const GenieProfile_Handle_t profileHandle) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(profileHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto configObj = genie::Embedding::Config::get(configHandle);
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
Genie_Status_t GenieEmbeddingConfig_bindLogger(const GenieEmbeddingConfig_Handle_t configHandle,
                                               const GenieLog_Handle_t logHandle) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(logHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto configObj = genie::Embedding::Config::get(configHandle);
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
Genie_Status_t GenieEmbeddingConfig_free(const GenieEmbeddingConfig_Handle_t configHandle) {
  try {
    GENIE_ENSURE(configHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    // Check if the embedding actually exists
    auto configObj = genie::Embedding::Config::get(configHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    configObj->unbindProfiler();
    configObj->unbindLogger();
    genie::Embedding::Config::remove(configHandle);
  } catch (const std::exception& e) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieEmbedding_create(const GenieEmbeddingConfig_Handle_t configHandle,
                                     GenieEmbedding_Handle_t* embeddingHandle) {
  try {
    const uint64_t startTime = genie::getTimeStampInUs();
    std::shared_ptr<ProfileStat> profileStat;
    GENIE_ENSURE(embeddingHandle, GENIE_STATUS_ERROR_INVALID_ARGUMENT);

    // Get config object
    auto configObj = genie::Embedding::Config::get(configHandle);
    GENIE_ENSURE(configObj, GENIE_STATUS_ERROR_INVALID_HANDLE);
    if (!configObj->getProfiler().empty())
      profileStat = std::make_shared<ProfileStat>(GENIE_PROFILE_EVENTTYPE_EMBEDDING_CREATE,
                                                  startTime,
                                                  "",
                                                  GENIE_PROFILE_COMPONENTTYPE_EMBEDDING);
    std::shared_ptr<genie::log::Logger> logger;
    if (!configObj->getLogger().empty())
      logger = *configObj->getLogger().begin();  // currently uses one logger
    else
      logger = nullptr;
    // Create embedding
    auto embedding = std::make_shared<genie::Embedding>(configObj, profileStat, logger);
    GENIE_ENSURE(embedding, GENIE_STATUS_ERROR_MEM_ALLOC);

    // Create Handle
    *embeddingHandle = genie::Embedding::add(embedding);

    embedding->bindProfiler(configObj->getProfiler());

    const uint64_t stopTime = genie::getTimeStampInUs();
    if (profileStat) {
      profileStat->setComponentId(embedding->getName().c_str());
      profileStat->setDuration(stopTime - startTime);
    }
    const std::unordered_set<std::shared_ptr<Profiler>> profiler = embedding->getProfiler();
    for (auto it : profiler) it->addProfileStat(profileStat);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }

  // Return SUCCESS
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieEmbedding_generate(const GenieEmbedding_Handle_t embeddingHandle,
                                       const char* queryStr,
                                       const GenieEmbedding_GenerateCallback_t callback,
                                       const void* userData) {
  int32_t status;

  try {
    const uint64_t startTime = genie::getTimeStampInUs();
    GENIE_ENSURE(embeddingHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    auto embedding = genie::Embedding::get(embeddingHandle);
    GENIE_ENSURE(embedding, GENIE_STATUS_ERROR_INVALID_HANDLE);
    std::shared_ptr<ProfileStat> profileStat;
    const std::unordered_set<std::shared_ptr<Profiler>> profiler = embedding->getProfiler();
    if (!profiler.empty())
      profileStat = std::make_shared<ProfileStat>(GENIE_PROFILE_EVENTTYPE_EMBEDDING_GENERATE,
                                                  startTime,
                                                  embedding->getName(),
                                                  GENIE_PROFILE_COMPONENTTYPE_EMBEDDING);
    GENIE_ENSURE(queryStr, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    GENIE_ENSURE(callback, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    status = embedding->generate(queryStr, callback, userData, profileStat);

    const uint64_t stopTime = genie::getTimeStampInUs();
    if (profileStat) profileStat->setDuration(stopTime - startTime);
    for (auto it : profiler) it->addProfileStat(profileStat);
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }

  return status;
}

GENIE_API
Genie_Status_t GenieEmbedding_free(const GenieEmbedding_Handle_t embeddingHandle) {
  try {
    const uint64_t startTime = genie::getTimeStampInUs();
    GENIE_ENSURE(embeddingHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    std::shared_ptr<ProfileStat> profileStat;
    std::unordered_set<std::shared_ptr<Profiler>> profiler;
    {
      // Check if the embedding actually exists
      auto embedding = genie::Embedding::get(embeddingHandle);
      GENIE_ENSURE(embedding, GENIE_STATUS_ERROR_INVALID_HANDLE);
      profiler = embedding->getProfiler();
      if (!profiler.empty())
        profileStat = std::make_shared<ProfileStat>(GENIE_PROFILE_EVENTTYPE_EMBEDDING_FREE,
                                                    startTime,
                                                    embedding->getName(),
                                                    GENIE_PROFILE_COMPONENTTYPE_EMBEDDING);
      embedding->unbindProfiler();
    }
    genie::Embedding::remove(embeddingHandle);

    const uint64_t stopTime = genie::getTimeStampInUs();
    if (profileStat) profileStat->setDuration(stopTime - startTime);
    for (auto it : profiler) it->addProfileStat(profileStat);
  } catch (const std::exception& e) {
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}
