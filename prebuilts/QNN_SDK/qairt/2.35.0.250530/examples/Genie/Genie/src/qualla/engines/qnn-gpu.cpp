//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")

#include <fmt/format.h>

#include <string>
#include <vector>

#include "gpu-model.hpp"
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

class GpuEngine : public Engine {
 private:
  QnnGpuModel::Params _params;
  std::unique_ptr<QnnGpuModel> _model;

 public:
  GpuEngine(Context& ctx, const qualla::json& json);
  ~GpuEngine();

  virtual size_t process(const std::vector<int32_t>& tokens,
                         std::vector<float>& logits,
                         bool logits_all) override;

  virtual size_t process(const std::vector<int32_t>& tokens,
                         Tensor& logits,
                         bool logits_all) override;

  virtual size_t process(const std::vector<int32_t>& tokens,
                         const std::vector<int32_t>& attention_map,
                         std::vector<float>& logits,
                         bool logits_all) override;

  virtual bool updateKV(size_t n_past) override;
  virtual bool save(const std::string& name) override;
  virtual size_t restore(const std::string& name, bool chooseHigherVariant) override;
  virtual void reset() override;

  virtual bool load() override;
  virtual bool unload() override;
};

namespace fs = std::filesystem;

GpuEngine::GpuEngine(Context& ctx, const qualla::json& json) : Engine(ctx, "qnn-gpu", json) {
  qualla::Timer start;

  using FF  = Feature::Flags;
  _features = FF::OUTPUT_LOGITS | FF::SAVE_RESTORE | FF::DYNAMIC_LOAD;

  __DEBUG("Qnn-Gpu : init start");

  qualla::Config conf(json, _type + "-engine:");

  if (conf.json.contains("longcontext")) {
    throw std::runtime_error("Long Context is not supported on GPU.");
  }

  // Parse config
  _params.model_basedir = conf.optional<std::string>("model-basedir", "");
  if (_params.model_basedir.is_relative()) {
    _params.model_basedir = _env.path().models / _params.model_basedir;
    _params.model_basedir = _params.model_basedir.make_preferred();
  }
  _params.model_list = conf.mandatory<std::vector<std::string>>("model-list");

  _params.ctx_size   = _ctx.size();
  _params.num_heads  = conf.optional<int64_t>("num-heads", 32);
  _params.head_dim   = conf.optional<int64_t>("head-dim", 128);
  _params.vocab_size = _ctx.n_vocab();

  if (!conf.optional<bool>("dynamic-load", false)) {
    load();
  }
};

GpuEngine::~GpuEngine() { unload(); }

bool GpuEngine::load() {
#ifdef _WIN32
  // QnnGpu Engine does not support Windows.
  return false;
#endif
  if (_model) return true;

  qualla::Timer start;
  bool status = true;

  __INFO("Qnn-Gpu : Loading Model");

  _model = std::make_unique<QnnGpuModel>(_env, _params);

  // Load model
  status = _model->initializeModel();
  if (!status) {
    throw std::runtime_error("Qnn-Gpu :Failure to initialize model");
  }

  // Initialize IO Tensor buffers
  status = _model->initializeIOTensors();
  if (!status) {
    throw std::runtime_error("Qnn-Gpu :Error in setting up IO Tensors");
  }

  // Initialize IO Tensor Pointers
  if (true != _model->initializeTensorPointers()) {
    throw std::runtime_error("Qnn-Gpu :Could not find I/O tensors in loaded graphs");
  }

  _kpis.load.update(start.elapsed_usec());
  return true;
}

bool GpuEngine::unload() {
  qualla::Timer start;
  __DEBUG("Qnn-Gpu : Unloading Model");
  _model.reset(nullptr);
  _kpis.unload.update(start.elapsed_usec());
  return true;
}

// KV Cache updation after each inference is handled inside QnnGpu Backend
// GPU Engine uses same memory handle for each KV input/output to the graph and uses
// Scatter op to update KV after each inference to the same memory handle.
bool GpuEngine::updateKV(size_t n_past) { return true; }

size_t GpuEngine::process(const std::vector<int32_t>& tokens,
                          std::vector<float>& logits,
                          bool logits_all) {
  return process(tokens, {}, logits, logits_all);
}

size_t GpuEngine::process(const std::vector<int32_t>& tokens,
                          const std::vector<int32_t>& attention_map,
                          std::vector<float>& logits,
                          bool logits_all) {
  if (!_model && !load()) {
    return 0;
  }
  qualla::Timer start;
  size_t n_tok = _model->runInference(tokens, attention_map, logits, logits_all);
  if (n_tok == 0) {
    State::error("Qnn-Gpu : RunInference Failed!");
  }
  _kpis.process.update(start.elapsed_usec());
  return n_tok;
}

size_t GpuEngine::process(const std::vector<int32_t>& tokens, Tensor& logits, bool logits_all) {
  if (!_model && !load()) {
    return 0;
  }
  qualla::Timer start;
  size_t n_tok = _model->runInference(tokens, logits, logits_all);
  if (n_tok == 0) {
    State::error("Qnn-Gpu : RunInference Failed!");
  }
  _kpis.process.update(start.elapsed_usec());
  return n_tok;
}

size_t GpuEngine::restore(const std::string& name, bool chooseHigherVariant) {
  if (!_model && !load()) {
    return 0;
  }

  fs::path cache_path = std::filesystem::path(name) / fmt::format("kv-cache.{}.qnn-gpu", _role);
  return _model->loadKVCache(cache_path.string());
}

bool GpuEngine::save(const std::string& name) {
  if (!_model && !load()) {
    return false;
  }

  fs::path cache_path = std::filesystem::path(name) / fmt::format("kv-cache.{}.qnn-gpu", _role);
  return _model->saveKVCache(cache_path.string());
}

// Reset requires clearing of KV caches only
void GpuEngine::reset() {
  if (!_model && !load()) {
    return;
  }
  _model->reset();
}

// Registrator instance
static OnLoad regy([]() {
  Engine::__register(
      "qnn-gpu", [](Context& ctx, const json& conf) { return (Engine*)new GpuEngine(ctx, conf); });
});

void needQnnGpuEngine() {}

}  // namespace qualla
