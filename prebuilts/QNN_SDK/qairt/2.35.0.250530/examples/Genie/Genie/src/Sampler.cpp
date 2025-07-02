//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#include <exception>
#include <set>

#include "Exception.hpp"
#include "Log.hpp"
#include "Macro.hpp"
#include "Sampler.hpp"
#include "qualla/detail/json.hpp"
#if ENABLE_DEBUG_LOGS
#include <iostream>
#endif

using namespace genie;

//=============================================================================
// Sampler functions
//=============================================================================

qnn::util::HandleManager<Sampler> Sampler::s_manager;

GenieSampler_Handle_t Sampler::add(std::shared_ptr<Sampler> config) {
  return (GenieSampler_Handle_t)s_manager.add(config);
}

std::shared_ptr<Sampler> Sampler::get(GenieSampler_Handle_t handle) {
  return s_manager.get((qnn::util::Handle_t)handle);
}

void Sampler::remove(GenieSampler_Handle_t handle) {
  s_manager.remove((qnn::util::Handle_t)handle);
}

Sampler::Sampler(qualla::json& origJson,
                 std::vector<std::reference_wrapper<qualla::Sampler>>& quallaSamplers)
    : m_origJson(origJson), m_quallaSamplers(quallaSamplers) {}

void Sampler::applyConfig(qualla::json samplerConfigJson) {
  std::string originalType =
      (m_origJson["sampler"]["type"] == nullptr ? "basic" : m_origJson["sampler"]["type"]);
  std::string type = originalType;
  if (samplerConfigJson["sampler"].contains("type")) {
    m_origJson["sampler"]["type"] = samplerConfigJson["sampler"]["type"];
    type                          = samplerConfigJson["sampler"]["type"];
  }

  if (type == "custom" && (samplerConfigJson["sampler"].contains("temp") ||
                           samplerConfigJson["sampler"].contains("top-k") ||
                           samplerConfigJson["sampler"].contains("top-p"))) {
    throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                    "Provided values are incompatible with custom sampler type.");
  }

  if (type == "basic" && samplerConfigJson["sampler"].contains("callback-name")) {
    throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                    "Provided values are incompatible with basic sampler type.");
  }
  if (samplerConfigJson["sampler"].contains("seed")) {
    m_origJson["sampler"]["seed"] = samplerConfigJson["sampler"]["seed"];
  }
  if (samplerConfigJson["sampler"].contains("temp")) {
    m_origJson["sampler"]["temp"] = samplerConfigJson["sampler"]["temp"];
  }
  if (samplerConfigJson["sampler"].contains("top-k")) {
    m_origJson["sampler"]["top-k"] = samplerConfigJson["sampler"]["top-k"];
  }
  if (samplerConfigJson["sampler"].contains("top-p")) {
    m_origJson["sampler"]["top-p"] = samplerConfigJson["sampler"]["top-p"];
  }
  if (samplerConfigJson["sampler"].contains("callback-name")) {
    m_origJson["sampler"]["callback-name"] = samplerConfigJson["sampler"]["callback-name"];
  }
  m_origJson["sampler"]["version"] =
      qualla::Config::optional<int32_t>(samplerConfigJson["sampler"], "version", 1);

#if ENABLE_DEBUG_LOGS
  std::cout << "Updated sampler config: " << std::endl;
  if (m_origJson["sampler"].contains("temp"))
    std::cout << "temp: " << m_origJson["sampler"]["temp"].get<double>() << std::endl;
  if (m_origJson["sampler"].contains("top-k"))
    std::cout << "top-k: " << m_origJson["sampler"]["top-k"] << std::endl;
  if (m_origJson["sampler"].contains("top-p"))
    std::cout << "top-p: " << m_origJson["sampler"]["top-p"].get<double>() << std::endl;
  if (m_origJson["sampler"].contains("seed"))
    std::cout << "seed: " << m_origJson["sampler"]["seed"] << std::endl;
  if (m_origJson["sampler"].contains("callback-name")) {
    std::cout << "callback-name: " << m_origJson["sampler"]["callback-name"] << std::endl;
  }
  std::cout << "type: " << m_origJson["sampler"]["type"] << std::endl;
#endif
  // Loop through the live qualla sampler instances and update the parameters
  for (auto& quallaSampler : m_quallaSamplers) {
    quallaSampler.get().applyConfig(m_origJson["sampler"]);
  }
}

void Sampler::registerCallback(const char* name, GenieSampler_ProcessCallback_t samplerCallback) {
  QNN_WARN("This API will soon be deprecated in favor of GenieSampler_registerUserDataCallback");
  std::string funcCbName = std::string(name);
  qualla::Sampler::registerProcessCallBack(funcCbName, samplerCallback);
}

void Sampler::registerUserDataCallback(const char* name, GenieSampler_UserDataCallback_t samplerCallback, const void* userData) {
  std::string funcCbName = std::string(name);
  qualla::Sampler::registerUserDataCallBack(funcCbName, samplerCallback, userData);
}

//=============================================================================
// Sampler::SamplerConfig functions
//=============================================================================

qnn::util::HandleManager<Sampler::SamplerConfig> Sampler::SamplerConfig::s_manager;

GenieSamplerConfig_Handle_t Sampler::SamplerConfig::add(
    std::shared_ptr<Sampler::SamplerConfig> config) {
  return (GenieSamplerConfig_Handle_t)s_manager.add(config);
}

std::shared_ptr<Sampler::SamplerConfig> Sampler::SamplerConfig::get(
    GenieSamplerConfig_Handle_t handle) {
  return s_manager.get((qnn::util::Handle_t)handle);
}

void Sampler::SamplerConfig::remove(GenieSamplerConfig_Handle_t handle) {
  s_manager.remove((qnn::util::Handle_t)handle);
}

Sampler::SamplerConfig::SamplerConfig(const char* configStr) {
  qualla::json quallaConfig;
  qualla::json config;
  {
    std::set<qualla::json> keys;

    auto callback = [&keys](int depth, qualla::json::parse_event_t event, qualla::json& parsed) {
      if ((depth == 1) && (event == qualla::json::parse_event_t::key)) {
        if (keys.count(parsed) > 0) {
          throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                          "Multiple sampler config key: " + parsed.dump());
        }
        keys.insert(parsed);
      }
      return true;
    };

    config = qualla::json::parse(configStr, callback);
  }

  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Sampler config is not an object");
  }

  if (!config.contains("sampler")) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing field: sampler");
  }

  // component is used in the "ENFORCE" macros
  const std::string component = "sampler";
  for (auto& item : config.items()) {
    if (item.key() == "sampler") {
      JSON_ENFORCE_OBJECT();
      validateSamplerConfig(item.value());
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown sampler config key: " + item.key());
    }
  }

  if (config["sampler"].contains("seed"))
    quallaConfig["sampler"]["seed"] = config["sampler"]["seed"];
  if (config["sampler"].contains("temp"))
    quallaConfig["sampler"]["temp"] = config["sampler"]["temp"];
  if (config["sampler"].contains("top-k"))
    quallaConfig["sampler"]["top-k"] = config["sampler"]["top-k"];
  if (config["sampler"].contains("top-p"))
    quallaConfig["sampler"]["top-p"] = config["sampler"]["top-p"];
  if (config["sampler"].contains("greedy"))
    quallaConfig["sampler"]["greedy"] = config["sampler"]["greedy"];
  if (config["sampler"].contains("version"))
    quallaConfig["sampler"]["version"] = config["sampler"]["version"];
  else
    quallaConfig["sampler"]["version"] = 1;
  if (config["sampler"].contains("type"))
    quallaConfig["sampler"]["type"] = config["sampler"]["type"];
  if (config["sampler"].contains("callback-name"))
    quallaConfig["sampler"]["callback-name"] = config["sampler"]["callback-name"];

  m_config = quallaConfig;
}

void Sampler::SamplerConfig::setParam(const std::string& keyStr, const std::string& valueStr) {
  if (!keyStr.empty()) {
    // Case 1: Only the parameter mentioned in keyStr is to be updated by valueStr
    std::set<std::string> validParams = {"seed", "top-p", "top-k", "temp", "type", "callback-name"};
    if (std::find(validParams.begin(), validParams.end(), keyStr) == validParams.end()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Invalid key obtained: " + keyStr);
    }
    try {
      if (keyStr == "seed")
        m_config["sampler"]["seed"] = std::stoi(valueStr);
      else if (keyStr == "top-p")
        m_config["sampler"]["top-p"] = std::stof(valueStr);
      else if (keyStr == "top-k")
        m_config["sampler"]["top-k"] = std::stof(valueStr);
      else if (keyStr == "temp")
        m_config["sampler"]["temp"] = std::stof(valueStr);
      else if (keyStr == "type")
        m_config["sampler"]["type"] = valueStr;
      else if (keyStr == "callback-name")
        m_config["sampler"]["callback-name"] = valueStr;
    } catch (const std::invalid_argument& e) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Invalid value obtained: " + valueStr + " for key: " + keyStr);
    }
  } else {
    // Case 2: User has passed entire json as a string in valueStr

    if (valueStr.empty())
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Both keyStr and valueStr cannot be empty");

    qualla::json config = qualla::json::parse(valueStr);
    if (!config.contains("sampler")) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing field: sampler");
    }

    // component is used in the "ENFORCE" macros
    const std::string component = "sampler";
    for (auto& item : config.items()) {
      if (item.key() == "sampler") {
        JSON_ENFORCE_OBJECT();
        validateSamplerConfig(item.value());
      } else {
        throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                        "Unknown sampler config key: " + item.key());
      }
    }

    m_config["sampler"]["seed"] =
        qualla::Config::optional<int32_t>(config["sampler"], "seed", m_config["sampler"]["seed"]);
    m_config["sampler"]["temp"] =
        qualla::Config::optional<float>(config["sampler"], "temp", m_config["sampler"]["temp"]);
    m_config["sampler"]["top-k"] =
        qualla::Config::optional<size_t>(config["sampler"], "top-k", m_config["sampler"]["top-k"]);
    m_config["sampler"]["top-p"] =
        qualla::Config::optional<float>(config["sampler"], "top-p", m_config["sampler"]["top-p"]);
    m_config["sampler"]["version"] = qualla::Config::optional<int32_t>(
        config["sampler"], "version", m_config["sampler"]["version"]);
    if (config["sampler"].contains("type")) m_config["sampler"]["type"] = config["sampler"]["type"];
    if (config["sampler"].contains("callback-name"))
      m_config["sampler"]["callback-name"] = config["sampler"]["callback-name"];
  }
}

void Sampler::SamplerConfig::validateSamplerConfig(const qualla::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "sampler config is not an object");
  }

  const std::set<std::string> mandatoryFields{"version"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing sampler field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  const std::string component = "sampler";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid sampler config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "seed") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "temp") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "top-k") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "top-p") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "greedy") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "type") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "callback-name") {
      JSON_ENFORCE_STRING();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown sampler config key: " + item.key());
    }
  }

  // For custom sampler, ensure type = "custom" and callback-name is specified
  if (config.contains("callback-name") && config.contains("type") && config["type"] != "custom") {
    throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                    "callback-name specified but type is set to: " + config["type"].dump() +
                        " Type must be custom");
  }

  if (config.contains("type") && config["type"] == "custom" && !config.contains("callback-name")) {
    throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                    "callback-name not specified but type is set to custom");
  }

  if ((config.contains("type") && config["type"] == "custom") &&
      (config.contains("temp") || config.contains("top-p") || config.contains("top-k") ||
       config.contains("greedy"))) {
    throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                    "Provided keys are not compatible with custom sampler type.");
  }
}

void Sampler::SamplerConfig::translateSamplerConfig(const qualla::json& genieConfig,
                                                    qualla::json& quallaConfig) {
  if (genieConfig["dialog"].contains("sampler")) {
    quallaConfig["sampler"]["type"] = "basic";

    if (genieConfig["dialog"]["sampler"].contains("seed")) {
      quallaConfig["sampler"]["seed"] = genieConfig["dialog"]["sampler"]["seed"];
    }
    if (genieConfig["dialog"]["sampler"].contains("temp")) {
      quallaConfig["sampler"]["temp"] = genieConfig["dialog"]["sampler"]["temp"];
    }
    if (genieConfig["dialog"]["sampler"].contains("type")) {
      quallaConfig["sampler"]["type"] = genieConfig["dialog"]["sampler"]["type"];
    }
    if (genieConfig["dialog"]["sampler"].contains("callback-name")) {
      quallaConfig["sampler"]["callback-name"] = genieConfig["dialog"]["sampler"]["callback-name"];
    }
    quallaConfig["sampler"]["role"] = "primary";
    if (genieConfig["dialog"]["type"] == "spd") {
      quallaConfig["sampler"]["role"] = "target";
    }

    if (genieConfig["dialog"]["sampler"].contains("top-k")) {
      quallaConfig["sampler"]["top-k"] = genieConfig["dialog"]["sampler"]["top-k"];
    }
    if (genieConfig["dialog"]["sampler"].contains("top-p")) {
      quallaConfig["sampler"]["top-p"] = genieConfig["dialog"]["sampler"]["top-p"];
    }
    if (genieConfig["dialog"]["sampler"].contains("greedy")) {
      quallaConfig["sampler"]["greedy"] = genieConfig["dialog"]["sampler"]["greedy"];
    }
    if (genieConfig["dialog"]["sampler"].contains("seed")) {
      quallaConfig["sampler"]["seed"] = genieConfig["dialog"]["sampler"]["seed"];
    }
  }
}

qualla::json Sampler::SamplerConfig::getJson() const { return m_config; }
