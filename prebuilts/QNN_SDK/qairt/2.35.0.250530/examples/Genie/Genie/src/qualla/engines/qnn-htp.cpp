//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "qnn-htp.hpp"

#define __ERROR(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_ERROR, fmt::format(__fmt, ##__VA_ARGS__))
#define __WARN(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_WARN, fmt::format(__fmt, ##__VA_ARGS__))
#define __INFO(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_INFO, fmt::format(__fmt, ##__VA_ARGS__))
#define __KPIS(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __DEBUG(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __TRACE(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __KVTRACE(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))

namespace qualla {

namespace fs = std::filesystem;

bool NspEngine::load() {
  if (_model) return true;

  qualla::Timer start;

  __INFO("qnn-htp: loading model");

  if (_model_type == "image") {
    _model = std::make_unique<QnnNspImageModel>(_env, _params);
  } else {
    _model = std::make_unique<QnnNspModel>(_env, _params);
  }

  // Load model
  if (true != _model->initializeModel()) {
    throw std::runtime_error("Failure to initialize model. " + _model->error());
  }

  // Initialize IO Tensor buffers
  if (true != _model->initializeIOTensors()) {
    throw std::runtime_error("Error in setting up IO Tensors");
  }

  if (true != _model->validateModel()) {
    throw std::runtime_error("Error validating model. Please check your I/O");
  }

  __INFO("qnn-htp: model has been validated!");

  if (true != _model->initializeKVManager(_params.n_threads, _params.cpumask, _params.poll)) {
    throw std::runtime_error("Error initializing KVCache managers: " + _model->error());
  }
  if (true != _model->initializeTensorPointers()) {
    throw std::runtime_error("Error : Could not find I/O tensors in loaded graphs");
  }

  if (true != _model->calculate_rope_embeddings()) {
    throw std::runtime_error("Error : Could not load precomputed position ids");
  }

  // Initialize LoRA
  if (_model->lora_conf == LoraConfigType::LORA_INPUT_WEIGHT_ENABLE) {
    if (true != _model->flushLoraWeightsBuffers())
      throw std::runtime_error("Error : Failed to flush the lora buffers");
  }

  if (true != _model->load_lmhead_weight_as_input()) {
    throw std::runtime_error("Error : Could not load lmhead weight input");
  }

  _kpis.load.update(start.elapsed_usec());
  return true;
}

bool NspEngine::unload() {
  qualla::Timer start;

  __DEBUG("qnn-htp: unloading model");
  _model.reset(nullptr);

  _kpis.unload.update(start.elapsed_usec());

  return true;
}

NspEngine::NspEngine(Context& ctx, const qualla::json& json) : Engine(ctx, "qnn-htp", json) {
  qualla::Timer start;

  using FF  = Feature::Flags;
  _features = FF::OUTPUT_LOGITS | FF::SAVE_RESTORE | FF::DYNAMIC_LOAD | FF::OUTPUT_EMBEDDINGS;

  __DEBUG("qnn-htp: init start");

  qualla::Config conf(json, _type + "-engine:");

  // Parse config
  _params.model_basedir = conf.optional<std::string>("model-basedir", "");
  if (_params.model_basedir.is_relative()) {
    _params.model_basedir = _env.path().models / _params.model_basedir;
    _params.model_basedir = _params.model_basedir.make_preferred();
  }
  _params.model_list = conf.mandatory<std::vector<std::string>>("model-list");

  // Parse model architecture
  std::string model_architecture = conf.optional<std::string>("model-architecture-type", "decoder");
  // Parse model architecture
  _model_type = conf.optional<std::string>("model-type", "text");

  if (model_architecture == "decoder")
    _params.modelArchitectureType = ModelArchitectureType::DECODER;
  else if (model_architecture == "encoder")
    _params.modelArchitectureType = ModelArchitectureType::ENCODER;
  else
    throw std::runtime_error(
        "Only Encoder and Decoder architectures are supported. Invalid architecture supplied : " +
        model_architecture);

  _params.backend_lib         = conf.optional<std::string>("backend-lib", "");
  _params.backend_ext_conf    = conf.optional<std::string>("backend-ext-conf", "");
  _params.draft_tok_map       = conf.optional<std::string>("draft-token-map", "");
  _params.ctx_size            = _ctx.size();
  _params.mmap_budget         = conf.optional<uint64_t>("mmap-budget", 0);
  _params.use_mmap            = conf.optional<bool>("use-mmap", true);
  _params.data_alignment_size = conf.optional<uint64_t>("data-alignment-size", 0);
  _params.use_async_Init      = conf.optional<bool>("use-async-Init", true);
  _params.spill_fill_bufsize  = conf.optional<int64_t>("spill-fill-bufsize", 0);
  _params.kv_dim              = conf.optional<int64_t>("kv-dim", 128);
  _params.n_embd              = _ctx.n_embd();
  _params.pad_token           = _ctx.pad();
  _params.variant_latency     = std::map<int, int>();
  _params.disable_kv_cache    = conf.optional<bool>("disable-kv-cache", false);
  _params.pooled_output       = conf.optional<bool>("pooled-output", true);
  _params.lmhead_weight_dir   = conf.optional<std::string>("lmhead-weight-dir", "");
  _params.graph_switching     = conf.optional<bool>("enable-graph-switching", false);
  _params.lazy_lora           = conf.optional<std::string>("graph-switching-lora-policy", "");
  _params.exec_select_graphs = conf.optional<std::vector<std::string>>("execute-select-graphs", {});
  _params.load_select_graphs = conf.optional<bool>("load-select-graphs", false);

  qualla::json latencies = conf.optional<qualla::json>("latency-map", {});
  for (auto& [variant, latency] : latencies.items())
    _params.variant_latency[std::stoi(variant)] = latency;
  _params.kv_update_method = conf.optional<std::string>("kv-update-method", "POINTER_SHIFT");
  _params.n_threads        = conf.optional<uint32_t>("n-threads", 4);
  if (_params.disable_kv_cache) {
    _params.n_threads = 0;
  }
  _params.poll = conf.optional<bool>("poll", false);

  // Positional encodings parameters
  if (conf.json.contains("positional-encoding")) {
    try {
      conf.json["positional-encoding"].get_to(_params.positional_encoding_params);
    } catch (const std::runtime_error& e) {
      State::fatal(fmt::format("Error in positional-encoding - {}", e.what()));
      throw std::runtime_error(State::error());
    }
  } else {  // For Backward compatibility. May be removed in future releases
    // __WARN("Using depracated positional encoding config. Please switch to positional-encoding");
    auto& pos_type = _params.positional_encoding_params;
    if (_params.modelArchitectureType == ModelArchitectureType::DECODER) {
      pos_type.type                     = PositionalEncoding::ROPE;
      pos_type.rope_params.dims         = conf.optional<int64_t>("pos-id-dim", 64);
      pos_type.rope_params.dims         = conf.optional("pos-id-dims", pos_type.rope_params.dims);
      pos_type.rope_params.theta        = conf.optional<double>("rope-theta", 10000.0);
      pos_type.rope_params.rope_scaling = conf.optional("rope-scaling", RopeScalingParams());
    } else {
      pos_type.type = PositionalEncoding::ABSOLUTE;
      // Other parameters for ENCODER ONLY model doesn't matter.
    }
  }
  // Default LoRA is Disabled
  uint8_t lora_version = conf.optional<uint8_t>("lora-version", 0);
  switch (lora_version) {
    case 0:
      _params.lora_config_type = LoraConfigType::LORA_DISABLE;
      break;
    case 1:
      _params.lora_config_type = LoraConfigType::LORA_INPUT_WEIGHT_ENABLE;
      break;
    case 2:
      _params.lora_config_type = LoraConfigType::LORA_ADAPTER_WEIGHT_ENABLE;
      break;
    default:
      throw std::runtime_error("Lora Verison Undefined.");
      break;
  }
  // LoRA adapter setting
  qualla::json lora_conf = conf.optional<qualla::json>("lora", {});
  if (lora_conf.size() != 0) {
    if (lora_conf.is_array()) {
      for (auto lc : lora_conf) {
        std::string lnm                           = lc["adapter-name"];
        _params.lora_param[lnm].lora_name         = lnm;
        _params.lora_param[lnm].alpha_tensor_name = lc["alpha-tensor-name"];
        std::vector<float> alphaInit{};
        uint32_t size = lc["alphas"].size();
        auto alpha    = lc["alphas"].get<std::vector<std::string>>();
        for (uint32_t i = 0; i < size; i++) {
          _params.lora_param[lnm].alphas.push_back(alpha[i]);
          alphaInit.push_back(1.0f);
        }
        _params.lora_param[lnm].alpha_tensor_val =
            (lc["alpha-tensor-value"].size() == size)
                ? lc["alpha-tensor-value"].get<std::vector<float>>()
                : alphaInit;

        if (_params.lora_config_type == LoraConfigType::LORA_ADAPTER_WEIGHT_ENABLE) {
          std::string basedir = "";
          if (lc.contains("binsection-basedir")) {
            basedir = lc["binsection-basedir"];
          }
          uint32_t n = lc["bin-sections"].size();
          for (uint32_t i = 0; i < n; i++) {
            auto binSec = lc["bin-sections"].get<std::vector<std::string>>();
            if (binSec[i].empty()) {
              _params.lora_param[lnm].binsection_list.push_back("");
              continue;
            }
            fs::path binsection_path = fs::path(binSec[i]);
            if (binsection_path.is_relative()) binsection_path = basedir / fs::path(binSec[i]);
            if (!fs::is_regular_file(binsection_path)) {
              __ERROR("qnn-htp: Can't access Lora binsection adapter : {}",
                      binsection_path.string());
              throw std::runtime_error("qnn-htp: Can't adapter file : " + binsection_path.string());
            }
            _params.lora_param[lnm].binsection_list.push_back(binsection_path.string());
          }
        } else if (_params.lora_config_type == LoraConfigType::LORA_INPUT_WEIGHT_ENABLE) {
          _params.lora_param[lnm].path = lc["path"];
        }
      }
    }
  }

  // Long context parameters
  if (conf.json.contains("longcontext")) {
    try {
      conf.json["longcontext"].get_to(_params.longcontext_params);
    } catch (const std::runtime_error& e) {
      State::fatal(fmt::format("Error in longcontext params - {}", e.what()));
      throw std::runtime_error(State::error());
    }
  }
  {
    qualla::json j = _params.longcontext_params;
    __DEBUG("LONGCTXDEBUG Long context parameters = {}", j.dump());
  }

  _params.embedding_length   = _ctx.embeddingLength();
  _params.embedding_datatype = _ctx.embeddingDatatype();
  // cpumask needs to be a string because JSON RFC doesn't allow for hex ints.
  std::string cpumask = conf.optional<std::string>("cpumask", "0");
  _params.cpumask     = std::stoull(cpumask, nullptr, 0);

  // Debug flags
  _params.debug_path    = conf.optional<std::string>("debug-path", "qualla_debug");
  _params.debug_specs   = conf.optional<bool>("debug-specs", false);
  _params.debug_tensors = conf.optional<bool>("debug-tensors", false);
  _params.debug_outputs = conf.optional<bool>("debug-outputs", false);
  _params.debug_qnn     = conf.optional<bool>("debug-qnn", static_cast<bool>(_env.logger()));

  if (!conf.optional<bool>("dynamic-load", false)) {
    load();
  }
};

NspEngine::~NspEngine() { unload(); }

bool NspEngine::updateKV(size_t n_past) { return updateKV(n_past, {}); }

bool NspEngine::updateKV(size_t n_past, const std::vector<bool>& selected) {
  if (!_model && !load()) return false;

  qualla::Timer start;

  if (n_past > _ctx.size()) {
    __ERROR("qnn-htp: context size exceeded : n_past {}", n_past);
    State::error("context size exceeded");
    return false;
  }

  if (!_model->setKVCacheNPast(n_past, selected)) {
    __ERROR("qnn-htp: Error updating KV$");
    return false;
  }

  __DEBUG("qnn-htp: Dispatched KV$ Update (n_past={}) in {} usec", n_past, start.elapsed_usec());

  _kpis.update_kv.update(start.elapsed_usec());

  return true;
}

size_t NspEngine::process(const std::vector<int32_t>& tokens,
                          std::vector<float>& logits,
                          bool logits_all) {
  return process(tokens, {}, logits, logits_all);
}

size_t NspEngine::process(const std::vector<int32_t>& tokens, Tensor& logits, bool logits_all) {
  return process(tokens, {}, logits, logits_all);
}

size_t NspEngine::process(const std::vector<int32_t>& tokens,
                          const std::vector<int32_t>& attention_map,
                          std::vector<float>& logits,
                          bool logits_all) {
  std::vector<uint8_t> unusedEmbeddings;
  const std::vector<int32_t> unusedTokens;
  const uint16_t* featureVector = nullptr;
  const std::vector<int32_t> selected;
  const int32_t start_idx = 0;
  const bool post_update  = false;
  return processAll(tokens,
                    unusedEmbeddings,
                    featureVector,
                    selected,
                    start_idx,
                    post_update,
                    attention_map,
                    logits,
                    logits_all);
}

size_t NspEngine::process(const std::vector<int32_t>& tokens,
                          const std::vector<int32_t>& attention_map,
                          Tensor& logits,
                          bool logits_all) {
  std::vector<uint8_t> unusedEmbeddings;
  const uint16_t* featureVector = nullptr;
  const std::vector<int32_t> selected;
  const int32_t start_idx = 0;
  const bool post_update  = false;
  return processAll(tokens,
                    unusedEmbeddings,
                    featureVector,
                    selected,
                    start_idx,
                    post_update,
                    attention_map,
                    logits,
                    logits_all);
}

size_t NspEngine::process(std::vector<uint8_t>& embeddings,
                          const std::vector<int32_t>& attention_map,
                          Tensor& logits,
                          bool logits_all) {
  const std::vector<int32_t> unusedTokens;
  const uint16_t* featureVector = nullptr;
  const std::vector<int32_t> selected;
  const int32_t start_idx = 0;
  const bool post_update  = false;
  return processAll(unusedTokens,
                    embeddings,
                    featureVector,
                    selected,
                    start_idx,
                    post_update,
                    attention_map,
                    logits,
                    logits_all);
}

size_t NspEngine::process(std::vector<uint8_t>& embedding_vectors,
                          const uint16_t* featureVector,
                          const std::vector<int32_t>& selected,
                          const int32_t start_idx,
                          const bool post_update,
                          const std::vector<int32_t>& attention_map,
                          Tensor& logits,
                          bool logits_all) {
  const std::vector<int32_t> unusedTokens;
  return processAll(unusedTokens,
                    embedding_vectors,
                    featureVector,
                    selected,
                    start_idx,
                    post_update,
                    attention_map,
                    logits,
                    logits_all);
}

size_t NspEngine::process(std::vector<uint8_t>& embedding_vectors,
                          const uint16_t* featureVector,
                          const std::vector<int32_t>& selected,
                          const int32_t start_idx,
                          const bool post_update,
                          const std::vector<int32_t>& attention_map,
                          std::vector<float>& logits,
                          bool logits_all) {
  const std::vector<int32_t> unusedTokens;
  return processAll(unusedTokens,
                    embedding_vectors,
                    featureVector,
                    selected,
                    start_idx,
                    post_update,
                    attention_map,
                    logits,
                    logits_all);
}

size_t NspEngine::process(std::vector<uint8_t>& embeddings,
                          const std::vector<int32_t>& attention_map,
                          std::vector<float>& logits,
                          bool logits_all) {
  const std::vector<int32_t> unusedTokens;
  const uint16_t* featureVector = nullptr;
  const std::vector<int32_t> selected;
  const int32_t start_idx = 0;
  const bool post_update  = false;
  return processAll(unusedTokens,
                    embeddings,
                    featureVector,
                    selected,
                    start_idx,
                    post_update,
                    attention_map,
                    logits,
                    logits_all);
}

size_t NspEngine::processAll(const std::vector<int32_t>& tokens,
                             std::vector<uint8_t>& embeddings,
                             const uint16_t* featureVector,
                             const std::vector<int32_t>& selected,
                             const int32_t start_idx,
                             const bool post_update,
                             const std::vector<int32_t>& attention_map,
                             std::vector<float>& logits,
                             bool logits_all) {
  if (!_model && !load()) return 0;
  qualla::Timer start;

  __DEBUG("qnn-htp: inference start: n_tokens {} ", embeddings.size());

  size_t n_tok = _model->runInference(tokens,
                                      embeddings,
                                      featureVector,
                                      selected,
                                      start_idx,
                                      post_update,
                                      attention_map,
                                      logits,
                                      logits_all);
  if (_model->failed()) {
    State::error(_model->error());
  }
  __DEBUG("qnn-htp: inference complete : {} usec", start.elapsed_usec());

  _kpis.process.update(start.elapsed_usec());

  return n_tok;
}

size_t NspEngine::processAll(const std::vector<int32_t>& tokens,
                             std::vector<uint8_t>& embeddings,
                             const uint16_t* featureVector,
                             const std::vector<int32_t>& selected,
                             const int32_t start_idx,
                             const bool post_update,
                             const std::vector<int32_t>& attention_map,
                             Tensor& logits,
                             bool logits_all) {
  if (!_model && !load()) return 0;
  qualla::Timer start;

  __DEBUG("qnn-htp: inference start: n_tokens {}", embeddings.size());

  size_t n_tok = _model->runInference(tokens,
                                      embeddings,
                                      featureVector,
                                      selected,
                                      start_idx,
                                      post_update,
                                      attention_map,
                                      logits,
                                      logits_all);
  if (_model->failed()) {
    State::error(_model->error());
  }
  __DEBUG("qnn-htp: inference complete : {} usec", start.elapsed_usec());

  _kpis.process.update(start.elapsed_usec());

  return n_tok;
}

size_t NspEngine::process(const std::vector<uint8_t>& inputs, std::vector<uint8_t>& outputs) {
  if (!_model && !load()) return 0;
  qualla::Timer start;

  size_t status = _model->runInference(inputs, outputs);
  if (status == 0) {
    State::error("qnn-htp : runInference failed!");
  }
  __DEBUG("qnn-htp: inference complete : {} usec", start.elapsed_usec());

  _kpis.process.update(start.elapsed_usec());

  return status;
}

bool NspEngine::cacheEosEmbedding(std::vector<uint8_t>& eosEmbedding) {
  if (!_model && !load()) {
    return false;
  }
  return _model->cacheEosEmbedding(eosEmbedding);
};

size_t NspEngine::getEmbeddingBufferSize() { return _model->getEmbeddingBufferSize(); }

bool NspEngine::set(qualla::json data) {
  bool ret = false;

  if (data.contains("kv-prefix-skip")) {
    _model->_size_to_skip_kv_prefix = data["kv-prefix-skip"].get<size_t>();
    ret                             = true;
  }

  if (data.contains("kv-prefix-offset")) {
    _model->_offset_to_apply_kv_prefix = data["kv-prefix-offset"].get<size_t>();
    ret                                = true;
  }
  return ret;
}

qualla::json NspEngine::get() {
  return {{"kv-prefix-skip", _model->_size_to_skip_kv_prefix},
          {"kv-prefix-offset", _model->_offset_to_apply_kv_prefix}};
}

qualla::InputType NspEngine::getInputType() { return _model->m_inputType; }

void NspEngine::getTensorParam(
    LayerType layerType, std::string& dataType, double& scale, int32_t& offset, size_t& bitWidth) {
  _model->getTensorParam(layerType, dataType, scale, offset, bitWidth);
}

void NspEngine::getTensorDimensions(LayerType layerType, std::vector<uint32_t>& dimensions) {
  _model->getTensorDimensions(layerType, dimensions);
}

size_t NspEngine::restore(const std::string& name, bool chooseHigherVariant) {
  if (!_model && !load()) return 0;

  if (_savedTokenCheckpoints.find(name) != _savedTokenCheckpoints.end())
    _tokensCheckpoint = _savedTokenCheckpoints[name];

  fs::path cache_path = std::filesystem::path(name) / fmt::format("kv-cache.{}.qnn-htp", _role);

  size_t ret = _model->loadKVCache(cache_path.string(), chooseHigherVariant);
  if (_model->failed()) State::error(_model->error());
  return ret;
}

bool NspEngine::save(const std::string& name) {
  if (!_model && !load()) return false;

  fs::path cache_path = std::filesystem::path(name) / fmt::format("kv-cache.{}.qnn-htp", _role);

  _savedTokenCheckpoints[name] = _tokensCheckpoint;

  bool ret = _model->saveKVCache(cache_path.string());
  if (_model->failed()) State::error(_model->error());
  return ret;
}

bool NspEngine::saveKvToBuffer(Buffer* kvBuff) {
  if (!_model && !load()) return false;

  bool ret = _model->saveKVCacheToBuffer(kvBuff);
  if (_model->failed()) State::error(_model->error());
  return ret;
}

bool NspEngine::getCacheSpec(CacheFileSpec& spec) {
  if (!_model && !load()) return false;

  bool ret = _model->getCacheSpec(spec);
  return ret;
}

bool NspEngine::getKVHead(
    CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) {
  if (!_model && !load()) return false;

  bool ret = _model->getKVHead(spec, layer, head, data, scale);
  return ret;
}

void NspEngine::reset() {
  if (!_model && !load()) return;

  // It's enough to just drop the KV$
  updateKV(0);
  _tokensCheckpoint.clear();
}

// Registrator instance
static OnLoad regy([]() {
  Engine::__register(
      "qnn-htp", [](Context& ctx, const json& conf) { return (Engine*)new NspEngine(ctx, conf); });
});
void needQnnHtpEngine() {}

bool NspEngine::applyLoraAdapter(std::string lora_adapter_name) {
  if (!_model) {
    __ERROR("qnn-htp: applyLoraAdapter failed model not initialized");
    return false;
  }
  if (_model->lora_conf == LoraConfigType::LORA_INPUT_WEIGHT_ENABLE) {
    return _model->applyLoraWeights(lora_adapter_name);
  } else
    return _model->applyLoraAdapter(lora_adapter_name);
}

bool NspEngine::applyLoraStrength(std::string tensor_name, float tensor_val) {
  if (!_model) {
    __ERROR("qnn-htp: applyLoraStrength failed model not initialized");
    return false;
  }
  return _model->applyLoraStrength(tensor_name, tensor_val);
}

bool NspEngine::updateTokenCheckpoint(uint32_t token, uint32_t kvCacheIndx) {
  if (!_model) {
    __ERROR("qnn-htp: updateTokenCheckpoint failed model not initialized");
    return false;
  }
  _tokensCheckpoint.push_back(std::make_pair(token, kvCacheIndx));
  return false;
}

bool NspEngine::removeTokenCheckpoint(size_t removeAmt) {
  if (!_model) {
    __ERROR("qnn-htp: removeTokenCheckpoint failed model not initialized");
    return false;
  }
  _tokensCheckpoint.erase(_tokensCheckpoint.end() - removeAmt, _tokensCheckpoint.end());
  return true;
}

std::pair<uint32_t, int32_t> NspEngine::rewindKVCacheToPrefixMatch(std::vector<int32_t>& tokens,
                                                                   uint32_t& past) {
  if (!_model) {
    __ERROR("qnn-htp: revertKVCacheToToken failed model not initialized");
    return {};
  }
  uint32_t idx         = 0;
  uint32_t last_n_past = 0;
  uint32_t rewindIndex = 0;
  uint32_t nextToken   = 0;

  for (int i = 0; i < _tokensCheckpoint.size() && idx < tokens.size(); i++) {
    if (_tokensCheckpoint[i].first != tokens[idx]) break;
    last_n_past = _tokensCheckpoint[i].second;
    rewindIndex = idx;
    if (i + 1 < _tokensCheckpoint.size())
      nextToken = _tokensCheckpoint[i + 1].first;
    else
      nextToken = -1;
    idx++;
  }
  updateKV(last_n_past + 1);
  if (_model) {
    _model->setHigherVariant();
  } else {
    return {};
  }
  past                   = last_n_past + 1;
  int lastCheckpointSize = _tokensCheckpoint.size();
  _tokensCheckpoint.resize(rewindIndex + 1);
  if (idx >= tokens.size() && idx <= lastCheckpointSize)
    return {rewindIndex + 1, nextToken};
  else
    return {rewindIndex + 1, -1};
}

bool NspEngine::setOemkey(const std::string& oemKey) {
  if (_model) {
    return _model->setOemKey(oemKey);
  }
  return false;
}
bool NspEngine::setExecutionPriority(const uint32_t executionPriority) {
  if (_model) {
    return _model->setExecutionPriority(executionPriority);
  }
  return false;
}

size_t NspEngine::getBuffer(void*& buffer, std::string bufferName, bool isPrompt) {
  size_t bufferSize = _model->getIOBufferByName(bufferName, buffer, isPrompt);
  return bufferSize;
}

std::string NspEngine::getTokenMapFilePath() {
  if (_model) return _model->m_draft_tok_map;
  return "";
}

bool NspEngine::isLongContextEnabled() const {
  return (_model) ? _model->isLongContextEnabled() : false;
}

}  // namespace qualla
