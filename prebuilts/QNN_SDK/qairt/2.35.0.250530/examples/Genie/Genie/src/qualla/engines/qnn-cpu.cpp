//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>

#include <string>
#include <vector>

#include "cpu-model.hpp"
#include "qualla/detail/config.hpp"
#include "qualla/detail/onload.hpp"
#include "qualla/detail/timer.hpp"
#include "qualla/engine.hpp"

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

class QnnCpuEngine : public Engine {
 private:
  // Model parameters
  std::unique_ptr<QnnCpuModel> _model;
  std::vector<std::pair<uint32_t, uint32_t>> m_tokensCheckpoint;

 public:
  QnnCpuEngine(Context& ctx, const qualla::json& json);
  ~QnnCpuEngine();

  virtual size_t process(const std::vector<int32_t>& tokens,
                         std::vector<float>& logits,
                         bool logits_all) override;

  virtual size_t process(const std::vector<int32_t>& tokens,
                         Tensor& logits,
                         bool logits_all) override;

  virtual size_t process(const std::vector<int32_t>& tokens,
                         const std::vector<int32_t>& attention_map,
                         Tensor& logits,
                         bool logits_all) override;

  virtual size_t process(std::vector<uint8_t>& embeddings,
                         const std::vector<int32_t>& attention_map,
                         Tensor& logits,
                         bool logits_all) override;

  virtual size_t getEmbeddingBufferSize() override;
  virtual bool updateKV(size_t n_past) override;
  virtual bool updateKV(size_t n_past, const std::vector<bool>& selected) override;
  virtual bool save(const std::string& name) override;
  virtual size_t restore(const std::string& name, bool chooseHigherVariant) override;
  virtual void reset() override;
  virtual bool applyLoraAdapter(std::string lora_adapter_name) override;
  virtual bool applyLoraStrength(std::string tensor_name, float tensor_val) override;
  virtual bool updateTokenCheckpoint(uint32_t token, uint32_t kvCacheIndx) override;
  virtual std::pair<uint32_t, int32_t> rewindKVCacheToPrefixMatch(std::vector<int32_t>& tokens,
                                                                  uint32_t& past) override;

  virtual qualla::InputType getInputType() override;
  virtual bool setKVHead(
      CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) override;
};

namespace fs = std::filesystem;

QnnCpuEngine::QnnCpuEngine(Context& ctx, const qualla::json& json) : Engine(ctx, "qnn-cpu", json) {
  qualla::Timer start;

  using FF  = Feature::Flags;
  _features = FF::OUTPUT_LOGITS | FF::SAVE_RESTORE | FF::OUTPUT_EMBEDDINGS;

  __DEBUG("qnn-cpu: init start");

  qualla::Config conf(json, _type + "-engine:");

  // Parse config
  QnnCpuModel::Params p;

  std::string model_input = conf.optional<std::string>("model-input", "tokens");
  if (model_input == "tokens")
    p.model_input = QnnCpuModel::ModelInput::TOKENS;
  else if (model_input == "embeddings")
    p.model_input = QnnCpuModel::ModelInput::INPUT_EMBEDDINGS;
  else
    throw std::runtime_error(
        "Only tokens and embeddings outputs are supported. Invalid output supplied : " +
        model_input);
  std::string model_output = conf.optional<std::string>("model-output", "logits");
  if (model_output == "logits")
    p.model_output = QnnCpuModel::ModelOutput::LOGITS;
  else if (model_output == "embeddings")
    p.model_output = QnnCpuModel::ModelOutput::EMBEDDINGS;
  else
    throw std::runtime_error(
        "Only logits and embeddings outputs are supported. Invalid output supplied : " +
        model_output);

  if (conf.json.contains("longcontext")) {
    throw std::runtime_error("Long Context is not supported on CPU.");
  }

  p.model_basedir        = _env.path().models / conf.optional<std::string>("model-basedir", "");
  p.model_bin_path       = conf.mandatory<std::string>("model-bin-path");
  p.model                = conf.mandatory<std::string>("model");
  p.op_package           = conf.mandatory<std::string>("op-package");
  p.backend_lib          = conf.mandatory<std::string>("backend-lib");
  p.n_threads            = conf.optional<uint32_t>("n-threads", 6);
  p.n_logits             = conf.optional<uint32_t>("n_logits", 1);
  p.n_layer              = conf.optional<uint32_t>("n_layer", 32);
  p.n_embd               = conf.optional<uint32_t>("n_embd", 4096);
  p.n_heads              = conf.optional<uint32_t>("n_heads", 32);
  p.n_kv_heads           = conf.optional<uint32_t>("n_kv_heads", 32);
  p.use_mmap             = conf.optional<bool>("use-mmap", false);
  p.kv_quant             = conf.optional<bool>("kv-quantization", false);
  p.ctx_size             = _ctx.size();
  p.n_vocab_size         = _ctx.n_vocab();
  p.embedding_datatype   = _ctx.embeddingDatatype();
  p.lora_config_type     = LoraConfigType::LORA_DISABLE;
  qualla::json lora_conf = conf.optional<qualla::json>("lora", {});
  if (lora_conf.size() != 0) {
    p.lora_config_type = LoraConfigType::LORA_ADAPTER_WEIGHT_ENABLE;
    if (lora_conf.is_array()) {
      for (auto lc : lora_conf) {
        std::string lnm                      = lc["adapter-name"];
        p.lora_config[lnm].lora_name         = lnm;
        p.lora_config[lnm].alpha_tensor_name = lc["alpha-tensor-name"];
        uint32_t size                        = lc["alphas"].size();
        auto alpha                           = lc["alphas"].get<std::vector<std::string>>();
        std::vector<float> alphaInit(size, 1.0f);
        for (uint32_t i = 0; i < size; i++) {
          p.lora_config[lnm].alphas.push_back(alpha[i]);
        }
        p.lora_config[lnm].alpha_tensor_val =
            (lc["alpha-tensor-value"].size() == size)
                ? lc["alpha-tensor-value"].get<std::vector<float>>()
                : alphaInit;

        std::string basedir = "";
        if (lc.contains("binsection-basedir")) {
          basedir = lc["binsection-basedir"];
        }
        uint32_t n = lc["bin-sections"].size();
        for (uint32_t i = 0; i < n; i++) {
          auto binSec              = lc["bin-sections"].get<std::vector<std::string>>();
          fs::path binsection_path = fs::path(binSec[i]);
          if (binsection_path.is_relative()) binsection_path = basedir / fs::path(binSec[i]);
          if (!fs::is_regular_file(binsection_path)) {
            __ERROR("qnn-cpu: Can't access Lora binsection adapter : {}", binsection_path.string());
            throw std::runtime_error("qnn-cpu: Can't open adapter file : " +
                                     binsection_path.string());
          }
          p.lora_config[lnm].binsection_list.push_back(binsection_path.string());
        }
      }
    }
  }

  _model = std::make_unique<QnnCpuModel>(_env, p);

  // Load model
  if (true != _model->initializeModel()) {
    throw std::runtime_error("Failure to initialize model");
  }

  // Initialize IO Tensor buffers
  if (true != _model->initializeIOTensors()) {
    throw std::runtime_error("Error in setting up IO Tensors");
  }

  if (true != _model->validateModel()) {
    throw std::runtime_error("Error validating model. Please check your I/O");
  }

  __DEBUG("qnn-cpu: model has been validated!");

  if (true != _model->initializeTensorPointers()) {
    throw std::runtime_error("Error : Could not find I/O tensors in loaded graphs");
  }

  _kpis.load.update(start.elapsed_usec());
};

QnnCpuEngine::~QnnCpuEngine() { __DEBUG("qnn-cpu: fini"); }

bool QnnCpuEngine::updateKV(size_t n_past) {
  qualla::Timer start;

  if (n_past > _ctx.size()) {
    __ERROR("qnn-cpu: context size exceeded : n_past {}", n_past);
    State::error("context size exceeded");
    return false;
  }

  __DEBUG("qnn-cpu: update-kv start : n_past {}", n_past);

  _model->setKVCacheNPast(n_past);

  __DEBUG("qnn-cpu: update-kv complete : {} usec", start.elapsed_usec());

  _kpis.update_kv.update(start.elapsed_usec());

  return true;
}

bool QnnCpuEngine::updateKV(size_t n_past, const std::vector<bool>& selected) {
  qualla::Timer start;

  if (n_past > _ctx.size()) {
    __ERROR("qnn-cpu: context size exceeded : n_past {}", n_past);
    State::error("context size exceeded");
    return false;
  }

  __DEBUG("qnn-cpu: update-kv start : n_past {}", n_past);

  _model->setKVCacheNPast(n_past);

  __DEBUG("qnn-cpu: update-kv complete : {} usec", start.elapsed_usec());

  _kpis.update_kv.update(start.elapsed_usec());

  return true;
}

size_t QnnCpuEngine::process(const std::vector<int32_t>& tokens,
                             std::vector<float>& logits,
                             bool logits_all = false) {
  qualla::Timer start;

  __DEBUG("qnn-cpu: inference start: n_tokens {}", tokens.size());

  _model->runInference(tokens, logits_all);

  __DEBUG("qnn-cpu: inference complete : {} usec", start.elapsed_usec());

  size_t n_tok;

  {
    qualla::Timer t;

    __DEBUG("qnn-cpu: get-logits start: all {}", logits_all);

    n_tok = _model->getDequantLogits(logits, logits_all);

    __DEBUG("qnn-cpu: get-logits complete : {} usec", t.elapsed_usec());
  }

  _kpis.process.update(start.elapsed_usec());

  return n_tok;
}

qualla::InputType QnnCpuEngine::getInputType() {
  return static_cast<qualla::InputType>(_model->model_input);
}

size_t QnnCpuEngine::process(const std::vector<int32_t>& tokens,
                             Tensor& logits,
                             bool logits_all = false) {
  qualla::Timer start;

  __DEBUG("qnn-cpu: inference start: n_tokens {}", tokens.size());

  _model->runInference(tokens, logits_all);

  __DEBUG("qnn-cpu: inference complete : {} usec", start.elapsed_usec());

  size_t n_tok;

  {
    qualla::Timer t;

    __DEBUG("qnn-cpu: get-logits start: all {}", logits_all);

    n_tok = _model->getLogits(logits, logits_all);

    __DEBUG("qnn-cpu: get-logits complete : {} usec", t.elapsed_usec());
  }

  _kpis.process.update(start.elapsed_usec());

  return n_tok;
}

size_t QnnCpuEngine::process(const std::vector<int32_t>& tokens,
                             const std::vector<int32_t>& attention_map,
                             Tensor& logits,
                             bool logits_all = false) {
  return process(tokens, logits, logits_all);
}

size_t QnnCpuEngine::process(std::vector<uint8_t>& embeddings,
                             const std::vector<int32_t>& attention_map,
                             Tensor& logits,
                             bool logits_all) {
  qualla::Timer start;

  __DEBUG("qnn-cpu: inference start: n_tokens {}", embeddings.size());

  _model->runInference(embeddings, logits_all);

  __DEBUG("qnn-cpu: inference complete : {} usec", start.elapsed_usec());

  size_t n_tok;

  {
    qualla::Timer t;

    __DEBUG("qnn-cpu: get-logits start: all {}", logits_all);

    n_tok = _model->getLogits(logits, logits_all);

    __DEBUG("qnn-cpu: get-logits complete : {} usec", t.elapsed_usec());
  }

  _kpis.process.update(start.elapsed_usec());

  return n_tok;
}

size_t QnnCpuEngine::getEmbeddingBufferSize() { return _model->getEmbeddingBufferSize(); }

size_t QnnCpuEngine::restore(const std::string& name, bool chooseHigherVariant) {
  fs::path cache_path = std::filesystem::path(name) / fmt::format("kv-cache.{}.qnn-cpu", _role);
  return _model->loadKVCache(cache_path.string());
}

bool QnnCpuEngine::setKVHead(
    CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) {
  bool ret = _model->setKVHead(spec, layer, head, data, scale);

  return ret;
}

bool QnnCpuEngine::save(const std::string& name) {
  fs::path cache_path = std::filesystem::path(name) / fmt::format("kv-cache.{}.qnn-cpu", _role);
  return _model->saveKVCache(cache_path.string());
}

void QnnCpuEngine::reset() {
  // It's enough to just drop the KV$
  updateKV(0);
  m_tokensCheckpoint.clear();
}

// For Lora
bool QnnCpuEngine::applyLoraAdapter(std::string lora_adapter_name) {
  if (!_model) {
    __ERROR("qnn-cpu: applyLoraAdapter failed, model not initialized");
    return false;
  }
  return _model->applyLoraAdapter(lora_adapter_name);
}

bool QnnCpuEngine::applyLoraStrength(std::string tensor_name, float tensor_val) {
  if (!_model) {
    __ERROR("qnn-cpu: applyLoraStrength failed, model not initialized");
    return false;
  }
  return _model->applyLoraStrength(tensor_name, tensor_val);
}

std::pair<uint32_t, int32_t> QnnCpuEngine::rewindKVCacheToPrefixMatch(std::vector<int32_t>& tokens,
                                                                      uint32_t& past) {
  if (!_model) {
    __ERROR("qnn-cpu: rewindKVCacheToPrefixMatch failed, model not initialized");
    return {};
  }
  uint32_t idx         = 0;
  uint32_t last_n_past = 0;
  uint32_t rewindIndex = 0;
  uint32_t nextToken   = 0;

  for (int i = 0; i < m_tokensCheckpoint.size() && idx < tokens.size(); i++) {
    if (m_tokensCheckpoint[i].first != tokens[idx]) break;
    last_n_past = m_tokensCheckpoint[i].second;
    rewindIndex = idx;
    if (i + 1 < m_tokensCheckpoint.size())
      nextToken = m_tokensCheckpoint[i + 1].first;
    else
      nextToken = -1;
    idx++;
  }

  updateKV(last_n_past + 1);
  past                   = last_n_past + 1;
  int lastCheckpointSize = m_tokensCheckpoint.size();
  m_tokensCheckpoint.resize(rewindIndex + 1);
  if (idx >= tokens.size() && idx <= lastCheckpointSize)
    return {rewindIndex + 1, nextToken};
  else
    return {rewindIndex + 1, -1};
}

bool QnnCpuEngine::updateTokenCheckpoint(uint32_t token, uint32_t kvCacheIndx) {
  m_tokensCheckpoint.push_back(std::make_pair(token, kvCacheIndx));
  return true;
}

// Registrator instance
static OnLoad regy([]() {
  Engine::__register("qnn-cpu", [](Context& ctx, const json& conf) {
    return (Engine*)new QnnCpuEngine(ctx, conf);
  });
});
void needQnnCpuEngine() {}

}  // namespace qualla
