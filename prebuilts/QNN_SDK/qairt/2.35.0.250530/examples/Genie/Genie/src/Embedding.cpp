//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <exception>
#include <fstream>
#include <set>
#include <sstream>

#include "Embedding.hpp"
#include "Exception.hpp"
#include "Macro.hpp"
#include "qualla/detail/json.hpp"
#include "qualla/env.hpp"

using namespace genie;

#ifdef _WIN32
inline std::string libPrefix = "";
inline std::string libSuffix = ".dll";
#else
inline std::string libPrefix = "lib";
inline std::string libSuffix = ".so";
#endif

inline std::string getLibName(std::string baseName) { return libPrefix + baseName + libSuffix; }

//=============================================================================
// Context::Config functions
//=============================================================================

static void validateContextConfig(const qualla::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "context config is not an object");
  }

  std::set<std::string> mandatoryFields{
      "version", "n-vocab", "ctx-size", "embed-size", "pad-token"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing context field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "context";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid context config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "n-vocab") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "ctx-size") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "embed-size") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "pad-token") {
      JSON_ENFORCE_NUMERIC();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown context config key: " + item.key());
    }
  }
}

static void translateContextConfig(const qualla::json& genieConfig, qualla::json& quallaConfig) {
  quallaConfig["n-vocab"]   = genieConfig["n-vocab"];
  quallaConfig["size"]      = genieConfig["ctx-size"];
  quallaConfig["n-embd"]    = genieConfig["embed-size"];
  quallaConfig["pad-token"] = genieConfig["pad-token"];
}

//=============================================================================
// Tokenizer::Config functions
//=============================================================================

static void validateTokenizerConfig(const qualla::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "tokenizer config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "path"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing tokenizer field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "tokenizer";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid tokenizer config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "path") {
      JSON_ENFORCE_STRING();
      // Note: the existence of this file is checked by qualla
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown tokenizer config key: " + item.key());
    }
  }
}

static void translateTokenizerConfig(const qualla::json& genieConfig, qualla::json& quallaConfig) {
  quallaConfig["tokenizer"] = genieConfig["path"];
}

//=============================================================================
// Backend::Config functions
//=============================================================================

static void validateBackendHtpConfig(const qualla::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "QnnHtp config is not an object");
  }

  std::set<std::string> mandatoryFields{
      "version", "spill-fill-bufsize", "use-mmap", "pooled-output", "allow-async-init"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing QnnHtp field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "QnnHtp";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid QnnHtp config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "spill-fill-bufsize") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "use-mmap") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "data-alignment-size") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "pooled-output") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "allow-async-init") {
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "disable-kv-cache") {
      JSON_ENFORCE_BOOLEAN();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown QnnHtp config key: " + item.key());
    }
  }
}

static void validateBackendGenaiConfig(const qualla::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "QnnGenAiTransformer config is not an object");
  }

  std::set<std::string> mandatoryFields{"version"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Missing QnnGenAiTransformer field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "QnnGenAiTransformer";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(
            GENIE_STATUS_ERROR_JSON_VALUE,
            "Invalid QnnGenAiTransformer config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "n-logits") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "n-layer") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "n-embd") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "n-heads") {
      JSON_ENFORCE_NUMERIC();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown QnnGenAiTransformer config key: " + item.key());
    }
  }
}

static void validateBackendConfig(const qualla::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "backend config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "type"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing backend field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "backend";

  std::string type;
  bool htp = false;
  qualla::json htpConfig;
  bool genai = false;
  qualla::json genaiConfig;

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid backend config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "type") {
      JSON_ENFORCE_STRING();
      type = item.value().get<std::string>();
      if (type == "QnnHtp") {
        htp = true;
      } else if (type == "QnnGenAiTransformer") {
        genai = true;
      } else {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid backend config: unsupported type: " + item.value().dump());
      }
    } else if (item.key() == "extensions") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "QnnHtp") {
      JSON_ENFORCE_OBJECT();
      htpConfig = item.value();
    } else if (item.key() == "QnnGenAiTransformer") {
      JSON_ENFORCE_OBJECT();
      genaiConfig = item.value();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown backend config key: " + item.key());
    }
  }

  if (htp) {
    if (!htpConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing QnnHtp embedding config");
    }
    validateBackendHtpConfig(htpConfig);
  } else {
    if (htpConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "QnnHtp backend config for incorrect backend type: " + type);
    }
  }

  if (genai) {
    if (!genaiConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Missing QnnGenAiTransformer embedding config");
    }
    validateBackendGenaiConfig(genaiConfig);
  } else {
    if (genaiConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "QnnGenAiTransformer backend config for incorrect backend type: " + type);
    }
  }
}

//=============================================================================
// Model::Config functions
//=============================================================================

static void validateLoraAdapterConfig(const qualla::json& config,
                                      LORA_VERSION& specifiedLoraVersion) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "lora adapter config is not an object");
  }
  const std::set<std::string> mandatoryFields{"version", "name"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing lora adapter field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  const std::string component        = "lora adapter";
  LORA_VERSION configuredLoraVersion = LORA_VERSION::GENIE_LORA_VERSION_UNDEFINED;
  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid lora config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "name") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "alphas") {
      JSON_ENFORCE_ARRAY();
      configuredLoraVersion = LORA_VERSION::GENIE_LORA_VERSION_V2;  // alphas occurs with V2 and V3
      for (auto& elem : item.value()) {
        if (!elem.is_string()) {
          throw Exception(GENIE_STATUS_ERROR_JSON_VALUE, "alphas must be an array of strings");
        }
      }
    } else if (item.key() == "bin-sections") {
      JSON_ENFORCE_ARRAY();
      configuredLoraVersion = LORA_VERSION::GENIE_LORA_VERSION_V2;  // Adapter occurs with V2 and V3
      for (auto& elem : item.value()) {
        if (!elem.is_string()) {
          throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                          "bin-sections must be an array of strings");
        }
      }
    } else if (item.key() == "path") {
      configuredLoraVersion = LORA_VERSION::GENIE_LORA_VERSION_V1;  // Weights are V1
      JSON_ENFORCE_STRING();
      // Note:all directory validations will done by NSP engine
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown lora adapter config key: " + item.key());
    }
  }

  if (specifiedLoraVersion == LORA_VERSION::GENIE_LORA_VERSION_V1 &&
      (configuredLoraVersion == LORA_VERSION::GENIE_LORA_VERSION_V2 ||
       configuredLoraVersion == LORA_VERSION::GENIE_LORA_VERSION_V3)) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                    "LoRA Adapters must be used with lora version: 2 or 3");
  } else if ((specifiedLoraVersion == LORA_VERSION::GENIE_LORA_VERSION_V2 ||
              specifiedLoraVersion == LORA_VERSION::GENIE_LORA_VERSION_V3) &&
             configuredLoraVersion == LORA_VERSION::GENIE_LORA_VERSION_V1) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                    "LoRA Weights must be used with lora version: 1");
  } else if (configuredLoraVersion == LORA_VERSION::GENIE_LORA_VERSION_UNDEFINED) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Invalid lora config.");
  }
}

static void validateLoraConfig(const qualla::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "lora config is not an object");
  }

  const std::set<std::string> mandatoryFields{"version", "adapters"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing lora field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  const std::string component       = "lora";
  LORA_VERSION specifiedLoraVersion = LORA_VERSION::GENIE_LORA_VERSION_V2;  // Default is loraV2
  if (config.find("lora-version") != config.end()) {
    switch (static_cast<uint8_t>(config["lora-version"])) {
      case 1:
        specifiedLoraVersion = LORA_VERSION::GENIE_LORA_VERSION_V1;
        break;
      case 2:
        specifiedLoraVersion = LORA_VERSION::GENIE_LORA_VERSION_V2;
        break;
      case 3:
        specifiedLoraVersion = LORA_VERSION::GENIE_LORA_VERSION_V3;
        break;
      default:
        specifiedLoraVersion = LORA_VERSION::GENIE_LORA_VERSION_UNDEFINED;
        break;
    }
  }

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid lora config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "alpha-tensor-name") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "adapters") {
      JSON_ENFORCE_ARRAY();
      for (auto& elem : item.value()) {
        validateLoraAdapterConfig(elem, specifiedLoraVersion);
      }
    } else if (item.key() == "lora-version") {  // Optional
      JSON_ENFORCE_NUMERIC();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown lora config key: " + item.key());
    }
  }
  if (specifiedLoraVersion == LORA_VERSION::GENIE_LORA_VERSION_UNDEFINED) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                    "Unsupported lora version: " + to_string(config["lora-version"]));
  }
}

static void validateModelBinaryConfig(const qualla::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "binary config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "ctx-bins"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing binary field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "binary";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid binary config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "ctx-bins") {
      JSON_ENFORCE_ARRAY();
      for (auto& elem : item.value()) {
        if (!elem.is_string()) {
          throw Exception(GENIE_STATUS_ERROR_JSON_VALUE, "ctx-bins must be an array of strings");
        }
      }
    } else if (item.key() == "lora") {
      JSON_ENFORCE_OBJECT();
      validateLoraConfig(item.value());
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown binary config key: " + item.key());
    }
  }
}

static void validateModelLibraryConfig(const qualla::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "library config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "model-bin"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing library field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "library";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid library config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "model-bin") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "lora") {
      JSON_ENFORCE_OBJECT();
      validateLoraConfig(item.value());
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown library config key: " + item.key());
    }
  }
}

static void validateModelConfig(const qualla::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "model config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "type"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing model field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "model";

  std::string type;
  bool binary = false;
  qualla::json binaryConfig;
  bool library = false;
  qualla::json libraryConfig;

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid model config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "type") {
      JSON_ENFORCE_STRING();
      type = item.value().get<std::string>();
      if (type == "binary") {
        binary = true;
      } else if (type == "library") {
        library = true;
      } else {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid model config: unsupported type: " + item.value().dump());
      }
    } else if (item.key() == "binary") {
      JSON_ENFORCE_OBJECT();
      binaryConfig = item.value();
    } else if (item.key() == "library") {
      JSON_ENFORCE_OBJECT();
      libraryConfig = item.value();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown model config key: " + item.key());
    }
  }

  if (binary) {
    if (!binaryConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing binary model config");
    }
    validateModelBinaryConfig(binaryConfig);
  } else {
    if (binaryConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "binary model config for incorrect model type: " + type);
    }
  }

  if (library) {
    if (!libraryConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing library model config");
    }
    validateModelLibraryConfig(libraryConfig);
  } else {
    if (libraryConfig.is_object()) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "library model config for incorrect model type: " + type);
    }
  }
}

//=============================================================================
// Engine::Config functions
//=============================================================================

static void validateEngineConfig(const qualla::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "engine config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "backend", "model"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing engine field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "engine";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid engine config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "backend") {
      JSON_ENFORCE_OBJECT();
      validateBackendConfig(item.value());
    } else if (item.key() == "mode") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "model") {
      JSON_ENFORCE_OBJECT();
      validateModelConfig(item.value());
    } else if (item.key() == "n-threads") {
      JSON_ENFORCE_NUMERIC();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown engine config key: " + item.key());
    }
  }
}

static void translateEngineConfig(const qualla::json& genieEngineConfig,
                                  qualla::json& quallaEngineConfig) {
  if (genieEngineConfig["version"] == 1) {
    if (genieEngineConfig.contains("n-threads"))
      quallaEngineConfig["n-threads"] = genieEngineConfig["n-threads"];

    if (genieEngineConfig["backend"]["type"] == "QnnHtp") {
      quallaEngineConfig["type"]                    = "qnn-htp";
      quallaEngineConfig["model-architecture-type"] = "encoder",
      quallaEngineConfig["backend-lib"]             = getLibName("QnnHtp");
      quallaEngineConfig["use-mmap"] = genieEngineConfig["backend"]["QnnHtp"]["use-mmap"];
      if (genieEngineConfig["backend"]["QnnHtp"].contains("data-alignment-size")) {
        quallaEngineConfig["data-alignment-size"] =
            genieEngineConfig["backend"]["QnnHtp"]["data-alignment-size"];
      }
      quallaEngineConfig["spill-fill-bufsize"] =
          genieEngineConfig["backend"]["QnnHtp"]["spill-fill-bufsize"];
      quallaEngineConfig["pooled-output"] = genieEngineConfig["backend"]["QnnHtp"]["pooled-output"];
      if (genieEngineConfig["backend"]["QnnHtp"].contains("disable-kv-cache")) {
        quallaEngineConfig["disable-kv-cache"] =
            genieEngineConfig["backend"]["QnnHtp"]["disable-kv-cache"];
      }
      if (genieEngineConfig.contains("mode")) {
        quallaEngineConfig["model-type"] = genieEngineConfig["mode"];
      }
      // By default, Qualla will default to the async init path.
      // For now, we are forcing async init off unless explicitly
      // specified in the Genie config. It is HTP specific feature only.
      quallaEngineConfig["use-async-Init"] = false;
      if (genieEngineConfig["backend"]["QnnHtp"].contains("allow-async-init")) {
        quallaEngineConfig["use-async-Init"] =
            genieEngineConfig["backend"]["QnnHtp"]["allow-async-init"];
      }
    } else if (genieEngineConfig["backend"]["type"] == "QnnGenAiTransformer") {
      quallaEngineConfig["type"]         = "qnn-cpu";
      quallaEngineConfig["model-output"] = "embeddings";
      quallaEngineConfig["backend-lib"]  = getLibName("QnnGenAiTransformer");
      if (genieEngineConfig["backend"]["QnnGenAiTransformer"].contains("n-logits")) {
        quallaEngineConfig["n_logits"] =
            genieEngineConfig["backend"]["QnnGenAiTransformer"]["n-logits"];
      }
      if (genieEngineConfig["backend"]["QnnGenAiTransformer"].contains("n-layer")) {
        quallaEngineConfig["n_layer"] =
            genieEngineConfig["backend"]["QnnGenAiTransformer"]["n-layer"];
      }
      if (genieEngineConfig["backend"]["QnnGenAiTransformer"].contains("n-embd")) {
        quallaEngineConfig["n_embd"] =
            genieEngineConfig["backend"]["QnnGenAiTransformer"]["n-embd"];
      }
      if (genieEngineConfig["backend"]["QnnGenAiTransformer"].contains("n-heads")) {
        quallaEngineConfig["n_heads"] =
            genieEngineConfig["backend"]["QnnGenAiTransformer"]["n-heads"];
      }
    }

    if (genieEngineConfig["backend"].contains("extensions")) {
      quallaEngineConfig["backend-ext-conf"] = genieEngineConfig["backend"]["extensions"];
    }

    if (genieEngineConfig["model"]["type"] == "binary") {
      quallaEngineConfig["model-list"] = genieEngineConfig["model"]["binary"]["ctx-bins"];
      if (genieEngineConfig["model"]["binary"].contains("lora")) {
        quallaEngineConfig["lora-version"] =
            static_cast<uint8_t>(LORA_VERSION::GENIE_LORA_VERSION_V2);
        if (genieEngineConfig["model"]["binary"]["lora"].contains("lora-version") &&
            genieEngineConfig["model"]["binary"]["lora"]["lora-version"] == 1) {
          quallaEngineConfig["lora-version"] =
              genieEngineConfig["model"]["binary"]["lora"]["lora-version"];
        }
        for (int i = 0; i < genieEngineConfig["model"]["binary"]["lora"]["adapters"].size(); i++) {
          quallaEngineConfig["lora"][i]["adapter-name"] =
              genieEngineConfig["model"]["binary"]["lora"]["adapters"][i]["name"];
          quallaEngineConfig["lora"][i]["alpha-tensor-name"] = "";
          if (genieEngineConfig["model"]["binary"]["lora"].contains("alpha-tensor-name")) {
            quallaEngineConfig["lora"][i]["alpha-tensor-name"] =
                genieEngineConfig["model"]["binary"]["lora"]["alpha-tensor-name"];
          }
          quallaEngineConfig["lora"][i]["alphas"] = qualla::json::array();
          if (genieEngineConfig["model"]["binary"]["lora"]["adapters"][i].contains("alphas")) {
            quallaEngineConfig["lora"][i]["alphas"] =
                genieEngineConfig["model"]["binary"]["lora"]["adapters"][i]["alphas"];
          } else {
            if (genieEngineConfig["model"]["binary"]["lora"].contains("alpha-tensor-name")) {
              quallaEngineConfig["lora"][i]["alphas"].emplace_back(
                  genieEngineConfig["model"]["binary"]["lora"]["alpha-tensor-name"]);
            }
          }
          quallaEngineConfig["lora"][i]["alpha-tensor-value"] = qualla::json::array();
          quallaEngineConfig["lora"][i]["binsection-basedir"] = "";
          if (genieEngineConfig["model"]["binary"]["lora"].contains("lora-version") &&
              genieEngineConfig["model"]["binary"]["lora"]["lora-version"] == 1) {
            quallaEngineConfig["lora"][i]["path"] =
                genieEngineConfig["model"]["binary"]["lora"]["adapters"][i]["path"];
          } else {
            quallaEngineConfig["lora"][i]["bin-sections"] =
                genieEngineConfig["model"]["binary"]["lora"]["adapters"][i]["bin-sections"];
          }
        }
      }
    } else if (genieEngineConfig["model"]["type"] == "library") {
      quallaEngineConfig["model"]          = getLibName("QnnGenAiTransformerModel");
      quallaEngineConfig["model-bin-path"] = genieEngineConfig["model"]["library"]["model-bin"];
      quallaEngineConfig["op-package"] =
          getLibName("QnnGenAiTransformerCpuOpPkg") + ":QnnOpPackage_interfaceProvider";
      if (genieEngineConfig["model"]["library"].contains("lora")) {
        for (int i = 0; i < genieEngineConfig["model"]["library"]["lora"]["adapters"].size(); i++) {
          quallaEngineConfig["lora"][i]["adapter-name"] =
              genieEngineConfig["model"]["library"]["lora"]["adapters"][i]["name"];
          if (genieEngineConfig["model"]["library"]["lora"].contains("alpha-tensor-name")) {
            quallaEngineConfig["lora"][i]["alpha-tensor-name"] =
                genieEngineConfig["model"]["library"]["lora"]["alpha-tensor-name"];
          }
          quallaEngineConfig["lora"][i]["alphas"] = qualla::json::array();
          if (genieEngineConfig["model"]["library"]["lora"]["adapters"][i].contains("alphas")) {
            quallaEngineConfig["lora"][i]["alphas"] =
                genieEngineConfig["model"]["library"]["lora"]["adapters"][i]["alphas"];
          } else {
            if (genieEngineConfig["model"]["library"]["lora"].contains("alpha-tensor-name")) {
              quallaEngineConfig["lora"][i]["alphas"].emplace_back(
                  genieEngineConfig["model"]["library"]["lora"]["alpha-tensor-name"]);
            }
          }
          quallaEngineConfig["lora"][i]["alpha-tensor-value"] = qualla::json::array();
          quallaEngineConfig["lora"][i]["binsection-basedir"] = "";
          quallaEngineConfig["lora"][i]["bin-sections"] =
              genieEngineConfig["model"]["library"]["lora"]["adapters"][i]["bin-sections"];
        }
      }
    }
  }
}

//=============================================================================
// Prompt::Config functions
//=============================================================================

static void validatePromptConfig(const qualla::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "prompt config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "prompt-template"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing prompt field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "prompt";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid context config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "prompt-template") {
      JSON_ENFORCE_ARRAY();
      for (auto& elem : item.value()) {
        if (!elem.is_string()) {
          throw Exception(GENIE_STATUS_ERROR_JSON_VALUE, "prompt tags must be an array of strings");
        }
      }
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Unknown context config key: " + item.key());
    }
  }
}

static void translatePromptConfig(const qualla::json& genieConfig, qualla::json& quallaConfig) {
  quallaConfig["tags"] = genieConfig["prompt-template"];
}

//=============================================================================
// Embedding::Config functions
//=============================================================================

qnn::util::HandleManager<Embedding::Config> Embedding::Config::s_manager;

GenieEmbeddingConfig_Handle_t Embedding::Config::add(std::shared_ptr<Embedding::Config> config) {
  return (GenieEmbeddingConfig_Handle_t)s_manager.add(config);
}

std::shared_ptr<Embedding::Config> Embedding::Config::get(GenieEmbeddingConfig_Handle_t handle) {
  return s_manager.get((qnn::util::Handle_t)handle);
}

void Embedding::Config::remove(GenieEmbeddingConfig_Handle_t handle) {
  s_manager.remove((qnn::util::Handle_t)handle);
}

static void validateLutConfig(const qualla::json& config) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "embedding config is not an object");
  }

  std::set<std::string> mandatoryFields{"version", "size"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing embedding field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "lut";
  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid embedding config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "size") {
      JSON_ENFORCE_NUMERIC();
    } else if (item.key() == "datatype") {
      JSON_ENFORCE_STRING();
      const std::set<std::string> supportedTypes = {
          "float32", "native", "ufixed8", "ufixed16", "sfixed8", "sfixed16"};
      if (std::find(supportedTypes.begin(), supportedTypes.end(), item.value()) ==
          supportedTypes.end()) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Unknown embedding datatype: " + std::string(item.value()));
      }
    } else if (item.key() == "lut-path") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "quant-param") {
      JSON_ENFORCE_OBJECT();
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown embedding config key: " + item.key());
    }
  }
}
void Embedding::validateEmbeddingConfig(const qualla::json& config, bool validateTextEncoder) {
  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Embedding config is not an object");
  }

  std::set<std::string> mandatoryFields{"version"};
  if (validateTextEncoder) {
    mandatoryFields.insert("context");
    mandatoryFields.insert("tokenizer");
    mandatoryFields.insert("engine");
  }
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing embedding field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "embedding";

  for (auto& item : config.items()) {
    if (item.key() == "version") {
      JSON_ENFORCE_NUMERIC();
      if (item.value().get<int>() != 1) {
        throw Exception(GENIE_STATUS_ERROR_JSON_VALUE,
                        "Invalid embedding config: unsupported version: " + item.value().dump());
      }
    } else if (item.key() == "type") {
      JSON_ENFORCE_STRING();
    } else if (item.key() == "context") {
      JSON_ENFORCE_OBJECT();
      validateContextConfig(item.value());
    } else if (item.key() == "tokenizer") {
      JSON_ENFORCE_OBJECT();
      validateTokenizerConfig(item.value());
    } else if (item.key() == "prompt") {  // optional parameter
      JSON_ENFORCE_OBJECT();
      validatePromptConfig(item.value());
    } else if (item.key() == "truncate-input") {  // optional parameter
      JSON_ENFORCE_BOOLEAN();
    } else if (item.key() == "engine") {
      JSON_ENFORCE_OBJECT();
      validateEngineConfig(config["engine"]);
    } else if (item.key() == "lut") {
      JSON_ENFORCE_OBJECT();
      validateLutConfig(config["lut"]);
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown embedding config key: " + item.key());
    }
  }
}

static void translateLutConfig(const qualla::json& lutConfig, qualla::json& quallaConfig) {
  quallaConfig["context"]["n-embd"] = lutConfig["size"];

  if (lutConfig.contains("datatype")) {
    std::string dataType = "QNN_DATATYPE_UNDEFINED";
    if (lutConfig["datatype"] == "float32") {
      dataType = "QNN_DATATYPE_FLOAT_32";
    } else if (lutConfig["datatype"] == "native") {
      dataType = "QNN_DATATYPE_UNDEFINED";
    } else if (lutConfig["datatype"] == "ufixed8") {
      dataType = "QNN_DATATYPE_UFIXED_POINT_8";
    } else if (lutConfig["datatype"] == "ufixed16") {
      dataType = "QNN_DATATYPE_UFIXED_POINT_16";
    } else if (lutConfig["datatype"] == "sfixed8") {
      dataType = "QNN_DATATYPE_SFIXED_POINT_8";
    } else if (lutConfig["datatype"] == "sfixed16") {
      dataType = "QNN_DATATYPE_SFIXED_POINT_16";
    }
    quallaConfig["context"]["embedding-datatype"] = dataType;
  }
  if (lutConfig.contains("quant-param")) {
    quallaConfig["context"]["quant-param"]["scale"]  = lutConfig["quant-param"]["scale"];
    quallaConfig["context"]["quant-param"]["offset"] = lutConfig["quant-param"]["offset"];
  }

  quallaConfig["lut-path"]  = lutConfig["lut-path"];
  quallaConfig["context"]   = quallaConfig["context"];
  quallaConfig["tokenizer"] = quallaConfig["tokenizer"];
}

void Embedding::translateEmbeddingConfig(const qualla::json& genieConfig,
                                         qualla::json& quallaConfig) {
  if (genieConfig.contains("context")) {
    translateContextConfig(genieConfig["context"], quallaConfig["context"]);
  }
  if (genieConfig.contains("prompt")) {
    translatePromptConfig(genieConfig["prompt"], quallaConfig["prompt"]);
  }
  if (genieConfig.contains("tokenizer")) {
    translateTokenizerConfig(genieConfig["tokenizer"], quallaConfig);
  }
  if (genieConfig.contains("engine")) {
    translateEngineConfig(genieConfig["engine"], quallaConfig["engine"]);
  }
  if (genieConfig.contains("type")) {
    quallaConfig["type"] = genieConfig["type"];
    if (genieConfig["type"] == "image-encoder") {
      quallaConfig["type"] = "ImageEncoder";
    }
  }
  if (genieConfig.contains("lut")) {
    translateLutConfig(genieConfig["lut"], quallaConfig);
  }
  if (genieConfig.contains(
          "truncate-input")) {  // to allow truncation of input incase it exceeds the context.
    quallaConfig["truncate-input"] = genieConfig["truncate-input"];
  }
}

Embedding::Config::Config(const char* configStr) {
  qualla::json config;

  {
    std::set<qualla::json> keys;

    auto callback = [&keys](int depth, qualla::json::parse_event_t event, qualla::json& parsed) {
      if ((depth == 1) && (event == qualla::json::parse_event_t::key)) {
        if (keys.count(parsed) > 0) {
          throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                          "Multiple embedding config key: " + parsed.dump());
        }
        keys.insert(parsed);
      }
      return true;
    };

    config = qualla::json::parse(configStr, callback);
  }

  if (!config.is_object()) {
    throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Embedding config is not an object");
  }

  std::set<std::string> mandatoryFields{"embedding"};
  for (const auto& field : mandatoryFields) {
    if (!config.contains(field)) {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA, "Missing embedding field: " + field);
    }
  }

  // component is used in the "ENFORCE" macros
  std::string component = "embedding";

  for (auto& item : config.items()) {
    if (item.key() == "embedding") {
      JSON_ENFORCE_OBJECT();
      bool validateTextEncoder = true;
      if (item.value().contains("type")) {
        if (item.value()["type"] == "image-encoder" || item.value()["type"] == "lut-encoder") {
          validateTextEncoder = false;
        }
      }
      validateEmbeddingConfig(item.value(), validateTextEncoder);
    } else {
      throw Exception(GENIE_STATUS_ERROR_JSON_SCHEMA,
                      "Unknown embedding config key: " + item.key());
    }
  }
  m_config = config;
}

qualla::json& Embedding::Config::getJson() { return m_config; }

void Embedding::Config::bindProfiler(std::shared_ptr<Profiler> profiler) {
  if (!profiler) return;
  profiler->incrementUseCount();
  m_profiler.insert(profiler);
}

void Embedding::Config::unbindProfiler() {
  for (auto it : m_profiler) it->decrementUseCount();
  m_profiler.clear();
}

std::unordered_set<std::shared_ptr<Profiler>>& Embedding::Config::getProfiler() {
  return m_profiler;
}

void Embedding::Config::bindLogger(std::shared_ptr<genie::log::Logger> logger) {
  if (!logger) return;
  logger->incrementUseCount();
  m_logger.insert(logger);
}

void Embedding::Config::unbindLogger() {
  for (auto it : m_logger) it->decrementUseCount();
  m_logger.clear();
}

std::unordered_set<std::shared_ptr<genie::log::Logger>>& Embedding::Config::getLogger() {
  return m_logger;
}

//=============================================================================
// Embedding functions
//=============================================================================

qnn::util::HandleManager<Embedding> Embedding::s_manager;
std::atomic<std::uint32_t> Embedding::s_nameCounter{0u};

GenieEmbedding_Handle_t Embedding::add(std::shared_ptr<Embedding> embedding) {
  return (GenieEmbedding_Handle_t)s_manager.add(embedding);
}

std::shared_ptr<Embedding> Embedding::get(GenieEmbedding_Handle_t handle) {
  return s_manager.get((qnn::util::Handle_t)handle);
}

void Embedding::remove(GenieEmbedding_Handle_t handle) {
  s_manager.remove((qnn::util::Handle_t)handle);
}

void Embedding::initEmbedding(qualla::json& config,
                              std::shared_ptr<ProfileStat> profileStat,
                              std::shared_ptr<genie::log::Logger> logger) {
  auto env = qualla::Env::create(qualla::json{});
  if (logger) env->bindLogger(logger);
  qualla::json quallaConfig;
  translateEmbeddingConfig(config["embedding"], quallaConfig);
  m_name            = "embedding" + std::to_string(s_nameCounter.fetch_add(1u));
  m_quallaEmbedding = qualla::Encoder::create(env, m_name, quallaConfig);
  if (!m_quallaEmbedding) {
    throw Exception(GENIE_STATUS_ERROR_MEM_ALLOC, "Could not create a embedding object");
  }

  qualla::Encoder::KPIs kpis = m_quallaEmbedding->kpis();
  if (profileStat)
    profileStat->translateKPIsToEvents(GENIE_PROFILE_EVENTTYPE_EMBEDDING_CREATE, kpis);
}

Embedding::Embedding(qualla::json& config,
                     std::shared_ptr<ProfileStat> profileStat,
                     std::shared_ptr<genie::log::Logger> logger) {
  initEmbedding(config, profileStat, logger);
}

Embedding::Embedding(std::shared_ptr<Config> config,
                     std::shared_ptr<ProfileStat> profileStat,
                     std::shared_ptr<genie::log::Logger> logger) {
  initEmbedding(config->getJson(), profileStat, logger);
}

void Embedding::bindProfiler(std::unordered_set<std::shared_ptr<Profiler>>& profiler) {
  for (auto it : profiler) {
    it->incrementUseCount();
    m_profiler.insert(it);
  }
}

void Embedding::unbindProfiler() {
  for (auto it : m_profiler) it->decrementUseCount();
  m_profiler.clear();
}

void Embedding::bindLogger(std::unordered_set<std::shared_ptr<genie::log::Logger>>& logger) {
  for (auto it : logger) {
    it->incrementUseCount();
    m_logger.insert(it);
    m_quallaEmbedding->getEnv()->bindLogger(it);
  }
}

void Embedding::unbindLogger() {
  for (auto it : m_logger) it->decrementUseCount();
  m_logger.clear();
}

std::unordered_set<std::shared_ptr<genie::log::Logger>>& Embedding::getLogger() { return m_logger; }

std::unordered_set<std::shared_ptr<Profiler>>& Embedding::getProfiler() { return m_profiler; }

std::string Embedding::getName() { return m_name; }

std::string Embedding::getType() { return m_type; }

int32_t Embedding::applyLora(std::string loraAdapterName, std::string engine) {
  bool status = m_quallaEmbedding->applyLoraAdapter(loraAdapterName, engine);
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_GENERAL);
}

int32_t Embedding::applyLoraStrength(std::string tensorName, std::string engine, float alpha) {
  bool status = m_quallaEmbedding->applyLoraStrength(tensorName, alpha, engine);
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_GENERAL);
}

bool Embedding::encode(const char* queryStr,
                       std::vector<uint8_t>& outputEmbedding,
                       std::shared_ptr<ProfileStat> profileStat) {
  std::string query(queryStr);
  bool status = m_quallaEmbedding->encode(query, outputEmbedding);
  if (status) {
    qualla::Encoder::KPIs kpis = m_quallaEmbedding->kpis();
    if (profileStat)
      profileStat->translateKPIsToEvents(GENIE_PROFILE_EVENTTYPE_EMBEDDING_GENERATE, kpis);
  }
  return status;
}

int32_t Embedding::generate(const char* queryStr,
                            GenieEmbedding_GenerateCallback_t callback,
                            const void* userData,
                            std::shared_ptr<ProfileStat> profileStat) {
  std::vector<uint8_t> outputEmbedding;
  bool status = false;
  status      = encode(queryStr, outputEmbedding, profileStat);
  if (status) {
    std::vector<uint32_t> dimensions;
    m_quallaEmbedding->output_dimensions(dimensions);
    callback(dimensions.data(), dimensions.size(), (float*)(outputEmbedding.data()), userData);
  }
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_GENERATE_FAILED);
}

int32_t Embedding::encode(const void* pixel_values,
                          const uint32_t inputSize,
                          std::vector<uint8_t>& outputEmbedding,
                          std::shared_ptr<ProfileStat> profileStat) {
  const uint8_t* embeddingsSrc = static_cast<const uint8_t*>(pixel_values);
  std::vector<uint8_t> embeddingVector(embeddingsSrc, embeddingsSrc + inputSize);
  bool status = false;
  status      = m_quallaEmbedding->encode(embeddingVector, outputEmbedding);
  return (status) ? (GENIE_STATUS_SUCCESS) : (GENIE_STATUS_ERROR_GENERATE_FAILED);
}

int32_t Embedding::getOutputDimensions(std::vector<uint32_t>& dimensions) {
  m_quallaEmbedding->output_dimensions(dimensions);
  return (GENIE_STATUS_SUCCESS);
}

int32_t Embedding::getOutputQuantParam(std::string& dataType,
                                       double& scale,
                                       int32_t& offset,
                                       size_t& byteWidth) {
  m_quallaEmbedding->outputTensorQuantParam(dataType, scale, offset, byteWidth);
  return (GENIE_STATUS_SUCCESS);
}