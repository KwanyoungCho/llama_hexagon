//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <filesystem>
#include <fstream>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "qualla/detail/cache-file.hpp"
#include "qualla/detail/config.hpp"
#include "qualla/detail/onload.hpp"
#include "qualla/detail/sampler-utils.hpp"
#include "qualla/detail/timer.hpp"
#include "qualla/dialog.hpp"
#include "qualla/sampler.hpp"

#define __ERROR(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_ERROR, fmt::format(__fmt, ##__VA_ARGS__))
#define __WARN(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_WARN, fmt::format(__fmt, ##__VA_ARGS__))
#define __INFO(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_INFO, fmt::format(__fmt, ##__VA_ARGS__))
#define __KPIS(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __DEBUG(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __TRACE(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __KVTRACE(__fmt, ...) \
  _LOG(_env->logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))

namespace fs = std::filesystem;
namespace qualla {

using qc = qualla::Config;
class KvShareDialog : public Dialog {
 public:
  KvShareDialog(std::shared_ptr<Env> env, const std::string& name, const json& conf);

  virtual bool process(std::vector<int32_t>& tokens, Dialog::Callback callback) override;

  virtual bool process(std::vector<int32_t>& tokens, qualla::DialogCallback callback) override;

  virtual void reset() override;

  // file-io implementation
  bool convertKV(const fs::path& cache_dir);

  size_t convertKV(Engine& p_engine, Engine& s_engine);

 private:
  int32_t _kv_quant;    // use quantization
  int32_t _use_in_mem;  // use buffer
};

KvShareDialog::KvShareDialog(std::shared_ptr<Env> env, const std::string& name, const json& conf)
    : Dialog(env, name, conf) {
  _kv_quant   = qc::optional<bool>(conf["engine"][1], "kv-quantization", false);
  _use_in_mem = qc::optional<bool>(conf["engine"][1], "use-in-memory", false);
}

void KvShareDialog::reset() {
  __INFO("dialog-reset: {}", _ctx->name());

  _n_past      = 0;
  _n_prompt    = 0;
  _n_generated = 0;
  _n_queries   = 0;
  _last_tok    = -1;

  _kpis.reset();

  // Reset Samplers
  for (auto& s : _sampler) s.second->reset();

  // Reset Engines
  for (auto& e : _engine) {
    e.second->reset();
  }

  State::clear();
}

bool KvShareDialog::process(std::vector<int32_t>& tokens, Dialog::Callback callback) {
  // Check for prev failures and bail out early
  if (State::failed()) return false;

  Timer start;

  // Vector for storing logits.
  // Allocated & filled by the engine.
  Tensor logits;

  State::clear();

  auto& sampler = *_sampler["primary"];

  auto& p_engine = *_engine["primary"];    // prompt
  auto& s_engine = *_engine["secondary"];  // generation

  if (_n_past + tokens.size() > _ctx->size()) {
    __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
    callback("", Sentence::END);
    return true;
  }

  if (!p_engine.process(tokens, logits))
    return Dialog::abort("engine prompt processing failed", callback);

  _n_prompt += tokens.size();
  _n_past += tokens.size();

  if (!p_engine.updateKV(_n_past)) return Dialog::abort("context size exceeded", callback);

  tokens[0] = _last_tok = sampler.process(logits);
  tokens.resize(1);

  _n_generated++;
  _kpis.prompt.update(start.elapsed_usec());
  // Log latest KPIs
  __KPIS("{}", kpis().dump(" "));

  start.reset();

  if (_ctx->is_eos(_last_tok)) {
    callback("", Sentence::END);
    return true;
  }

  if (!callback(_tokenizer->decode(tokens), Sentence::BEGIN)) return true;

  size_t n = 0;
  if (!_use_in_mem) {
    __DEBUG("dialog: {} : switching engines", _ctx->name());
    {
      // Setup cache dir for saving the engine state
      std::string cache_name = _ctx->name() + "-kv-share";
      fs::path cache_dir     = _env->path().cache / cache_name;

      if (!fs::exists(cache_dir) && !fs::create_directories(cache_dir)) {
        __ERROR(
            "dialog: {} : failed to create cache directory {}", _ctx->name(), cache_dir.string());
        return Dialog::abort("engine switch failed", callback);
      }

      // Save the primary engine
      p_engine.save(cache_name);

      // The purpose is to save the hyperparams
      s_engine.save(cache_name);

      convertKV(cache_dir);

      n = s_engine.restore(cache_name);

      if (!fs::remove_all(cache_dir)) {
        __WARN("dialog: {} : cache files not closed/dir not found", _ctx->name());
      }
    }
  } else {
    n = convertKV(p_engine, s_engine);
  }

  if (n != _n_past) {
    __WARN("dialog: {} : kv size mismatch {} expected {}", _ctx->name(), n, _n_past);
    _n_past = n;
  }
  s_engine.updateKV(_n_past);

  State::busy(true);

  while (true) {
    if (State::canceled()) {
      callback("", Sentence::END);
      break;
    }

    if (_n_past + tokens.size() > _ctx->size()) {
      __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
      callback("", Sentence::END);
      break;
    }
    if (!s_engine.process(tokens, logits))
      return Dialog::abort("secondary engine processing failed", callback);

    tokens[0] = _last_tok = sampler.process(logits);

    _n_past++;
    _n_generated++;

    if (!s_engine.updateKV(_n_past)) return Dialog::abort("context size exceeded", callback);

    if (_ctx->is_eos(_last_tok)) {
      callback("", Sentence::END);
      break;
    }

    if (!callback(_tokenizer->decode(tokens), Sentence::CONTINUE)) break;
  }

  State::busy(false);

  _kpis.generate.update(start.elapsed_usec());

  // Log latest KPIs in a single line
  __KPIS("{}", kpis().dump(" "));

  return true;
}

bool KvShareDialog::process(std::vector<int32_t>& tokens, qualla::DialogCallback callback) {
  // Check for prev failures and bail out early
  if (State::failed()) return false;

  Timer start;

  // Vector for storing logits.
  // Allocated & filled by the engine.
  Tensor logits;

  State::clear();

  auto& sampler = *_sampler["primary"];

  auto& p_engine = *_engine["primary"];    // prompt
  auto& s_engine = *_engine["secondary"];  // generation

  if (_n_past + tokens.size() > _ctx->size()) {
    __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
    callback.callBack(nullptr, 0, Sentence::END, tokenizer());
    return true;
  }

  if (!p_engine.process(tokens, logits))
    return Dialog::abort("engine prompt processing failed", callback);

  _n_prompt += tokens.size();
  _n_past += tokens.size();

  if (!p_engine.updateKV(_n_past)) return Dialog::abort("context size exceeded", callback);

  tokens[0] = _last_tok = sampler.process(logits);
  tokens.resize(1);

  _n_generated++;

  _kpis.prompt.update(start.elapsed_usec());
  // Log latest KPIs
  __KPIS("{}", kpis().dump(" "));

  start.reset();

  if (_ctx->is_eos(_last_tok)) {
    callback.callBack(nullptr, 0, Sentence::END, tokenizer());
    return true;
  }

  if (!callback.callBack(tokens.data(), tokens.size(), Sentence::BEGIN, tokenizer())) return true;

  size_t n = 0;
  if (!_use_in_mem) {
    __DEBUG("dialog: {} : switching engines", _ctx->name());
    {
      // Setup cache dir for saving the engine state
      std::string cache_name = _ctx->name() + "-kv-share";
      fs::path cache_dir     = _env->path().cache / cache_name;

      if (!fs::exists(cache_dir) && !fs::create_directories(cache_dir)) {
        __ERROR(
            "dialog: {} : failed to create cache directory {}", _ctx->name(), cache_dir.string());
        return Dialog::abort("engine switch failed", callback);
      }

      // Save the primary engine
      p_engine.save(cache_name);

      // The purpose is to save the hyperparams
      s_engine.save(cache_name);

      convertKV(cache_dir);

      n = s_engine.restore(cache_name);

      if (!fs::remove_all(cache_dir)) {
        __WARN("dialog: {} : cache files not closed/dir not found", _ctx->name());
      }
    }
  } else {
    n = convertKV(p_engine, s_engine);
  }

  if (n != _n_past) {
    __WARN("dialog: {} : kv size mismatch {} expected {}", _ctx->name(), n, _n_past);
    _n_past = n;
  }
  s_engine.updateKV(_n_past);

  State::busy(true);

  while (true) {
    if (State::canceled()) {
      callback.callBack(nullptr, 0, Sentence::END, tokenizer());
      break;
    }

    if (_n_past + tokens.size() > _ctx->size()) {
      __WARN("Context limit exceeded ({} + 1 > {})", _n_past, _ctx->size());
      callback.callBack(nullptr, 0, Sentence::END, tokenizer());
      break;
    }
    if (!s_engine.process(tokens, logits))
      return Dialog::abort("secondary engine processing failed", callback);

    tokens[0] = _last_tok = sampler.process(logits);

    _n_past++;
    _n_generated++;

    if (!s_engine.updateKV(_n_past)) return Dialog::abort("context size exceeded", callback);

    if (_ctx->is_eos(_last_tok)) {
      callback.callBack(nullptr, 0, Sentence::END, tokenizer());
      break;
    }

    if (!callback.callBack(tokens.data(), tokens.size(), Sentence::CONTINUE, tokenizer())) break;
  }

  State::busy(false);

  _kpis.generate.update(start.elapsed_usec());

  // Log latest KPIs in a single line
  __KPIS("{}", kpis().dump(" "));

  return true;
}

void convertKVLayers(qualla::Engine* p_engine,
                     qualla::Engine* s_engine,
                     CacheFileSpec spec,
                     uint32_t layer,
                     uint32_t len,
                     std::shared_ptr<Env>& _env) {
  uint32_t kv_dim = spec.embed_dim;
  uint32_t n_tok  = spec.update_size;
  uint32_t n_head = spec.n_heads;

  std::vector<uint8_t> head_buffer(2 * n_tok * kv_dim);
  std::vector<double> kv_scales(2);

  for (uint32_t i = layer; i < (layer + len); i++) {
    for (uint32_t j = 0; j < n_head; j++) {
      if (!p_engine->getKVHead(spec, i, j, head_buffer.data(), kv_scales.data())) {
        __ERROR("kv-convert: could not fetch head {} of layer {}", j, i);
      }

      if (!s_engine->setKVHead(spec, i, j, head_buffer.data(), kv_scales.data())) {
        __ERROR("kv-convert: could not set head {} of layer {}", j, i);
      }
    }
  }
}

size_t KvShareDialog::convertKV(qualla::Engine& p_engine, qualla::Engine& s_engine) {
  Timer start;

  CacheFileSpec spec;
  p_engine.getCacheSpec(spec);

  __DEBUG(
      "kv-convert: load {{ num_tensors {}, magic {}, dtype {}, n_heads {}, embed_dim {} "
      "update_size {} }}",
      spec.num_tensors,
      spec.magic,
      int(spec.dtype),
      spec.n_heads,
      spec.embed_dim,
      spec.update_size);

  uint32_t n_layer = spec.num_tensors / 2;
  uint32_t n_tok   = spec.update_size;

  std::vector<std::thread> convert_threads;
  uint32_t num_threads = (std::thread::hardware_concurrency() * 2) / 3;
  uint32_t len         = std::ceil(n_layer / (float)num_threads);
  for (uint32_t i = 0; i < len * num_threads; i += len) {
    if (i < n_layer) {
      uint32_t process = ((n_layer - i) < len) ? (n_layer - i) : len;
      convert_threads.emplace_back(
          convertKVLayers, &p_engine, &s_engine, spec, i, process, std::ref(_env));
    } else {
      break;
    }
  }

  // Wait for all key threads to finish
  for (auto& t : convert_threads) {
    t.join();
  }

  __DEBUG("kv-convert: done converting in {} usec", start.elapsed_usec());

  return (size_t)n_tok;
}

bool KvShareDialog::convertKV(const fs::path& cache_dir) {
  Timer start;

  fs::path nsp_cache_path = cache_dir / "kv-cache.primary.qnn-htp";
  fs::path cpu_cache_path = cache_dir / "kv-cache.secondary.qnn-cpu";

  __DEBUG("kv-convert: begin converting {} to ", nsp_cache_path.string(), cpu_cache_path.string());

  std::ifstream nsp_fs(nsp_cache_path, std::ios::in | std::ios::binary);

  if (nsp_fs.fail()) {
    __ERROR("kv-convert: error reading file {}", nsp_cache_path.string());
    State::error("failed to read primary kv-cache");
    return false;
  }

  // Read spec from nsp file
  CacheFileSpec nsp_spec;
  nsp_fs.read((char*)&nsp_spec, sizeof(nsp_spec));
  if (nsp_spec.magic != 0xC0DE) {
    __ERROR("kv-convert: expected 0xC0DE found {:#x}", nsp_spec.magic);
    State::error("invalid format of primary kv-cache");
    return false;
  }

  __DEBUG(
      "kv-convert: load {{ num_tensors {}, magic {}, dtype {}, n_heads {}, embed_dim {} "
      "update_size {} }}",
      nsp_spec.num_tensors,
      nsp_spec.magic,
      int(nsp_spec.dtype),
      nsp_spec.n_heads,
      nsp_spec.embed_dim,
      nsp_spec.update_size);

  std::fstream cpu_fs(cpu_cache_path, std::ios::in | std::ios::out | std::ios::binary);

  if (cpu_fs.fail()) {
    // TODO: replace with proper error handling
    __ERROR("kv-convert: failed to write {}", cpu_cache_path.string());
    State::error("failed to save secondary kv-cache");
    return false;
  }

  CacheFileSpec cpu_spec;
  cpu_fs.read((char*)&cpu_spec, sizeof(cpu_spec));
  if (cpu_spec.magic != 0xC0DE) {
    __ERROR("kv-convert: expected 0xC0DE found {:#x}", cpu_spec.magic);
    State::error("invalid format of secondary kv-cache");
    return false;
  }

  // Set the n_tokens processed during prompt processing and the spec write to file
  cpu_spec.update_size = nsp_spec.update_size;
  cpu_fs.seekp(std::ios::beg);
  cpu_fs.write((char*)&cpu_spec, sizeof(cpu_spec));

  const uint32_t n_layer = nsp_spec.num_tensors / 2;
  const uint32_t n_head  = nsp_spec.n_heads;
  const uint32_t kv_dim  = nsp_spec.embed_dim;
  const uint32_t n_tok   = nsp_spec.update_size;

  const size_t cache_size = n_layer * n_head * kv_dim * n_tok;

  // Read Key/Value Cache
  std::vector<uint8_t> key_cache(cache_size);
  std::vector<uint8_t> value_cache(cache_size);
  nsp_fs.read((char*)key_cache.data(), cache_size);
  nsp_fs.read((char*)value_cache.data(), cache_size);

  // Read Quantization parameters
  std::vector<double> key_scales(n_layer);
  std::vector<double> value_scales(n_layer);
  nsp_fs.read((char*)key_scales.data(), n_layer * sizeof(double));
  nsp_fs.read((char*)value_scales.data(), n_layer * sizeof(double));

  nsp_fs.close();

  // Convert and write on cpu_file
  // Dequant and transpose caches
  const uint32_t layer_size = n_head * kv_dim * n_tok;
  const uint32_t head_size  = kv_dim * n_tok;

  // Transpose kvdim * n_tok (QNN-HTP K$) -> n_tok * kvdim (QNN-CPU K$)
  // For ScopGPT KV$ Format
  __DEBUG("kv-convert: dequantizing keys");
  std::vector<float> dequant_keys(cache_size);
  for (uint32_t i = 0; i < n_layer; i++) {
    for (uint32_t j = 0; j < n_head; j++) {
      for (uint32_t k = 0; k < kv_dim; k++) {
        for (uint32_t l = 0; l < n_tok; l++) {
          // Interleave K$
          // QNN HTP: [0 2 4 ... 126 1 3 5 ... 127]
          // QNN CPU: [0 1 2 ... 63  64 65 ... 127]
          const uint32_t interleaved_k = (2 * k < kv_dim) ? 2 * k : 2 * (k - kv_dim / 2) + 1;

          const uint32_t read_loc  = i * layer_size + j * head_size + k * n_tok + l;
          const uint32_t write_loc = i * layer_size + j * head_size + l * kv_dim + interleaved_k;

          dequant_keys[write_loc] = (static_cast<float>(key_cache[read_loc]) - 128) * key_scales[i];
        }
      }
    }
  }

  __DEBUG("kv-convert: dequantizing values");
  std::vector<float> dequant_values(cache_size);
  for (uint32_t i = 0; i < n_layer; i++) {
    for (uint32_t j = 0; j < n_head; j++) {
      for (uint32_t l = 0; l < n_tok; l++) {
        for (uint32_t k = 0; k < kv_dim; k++) {
          const uint32_t read_loc  = i * layer_size + j * head_size + l * kv_dim + k;
          const uint32_t write_loc = read_loc;

          dequant_values[write_loc] =
              (static_cast<float>(value_cache[read_loc]) - 128) * value_scales[i];
        }
      }
    }
  }

  // Q8_0_32 quantization
  if (_kv_quant) {
    const uint32_t block_size = 32;
    const uint32_t num_blocks = n_layer * n_head * n_tok * (kv_dim / block_size);

    std::vector<int8_t> q8_keys_quant(cache_size);
    std::vector<float> q8_keys_scales(num_blocks);

    std::vector<int8_t> q8_values_quant(cache_size);
    std::vector<float> q8_values_scales(num_blocks);
    // call scale // each block has different scale

    for (uint32_t i = 0; i < n_layer; i++) {
      for (uint32_t j = 0; j < n_head; j++) {
        for (uint32_t l = 0; l < n_tok; l++) {
#pragma clang loop vectorize(enable)
          for (uint32_t k = 0; k < kv_dim / block_size; k++) {
            const uint32_t quant_loc = i * layer_size + j * head_size + l * kv_dim;
            const uint32_t key_loc = i * (layer_size / block_size) + j * (head_size / block_size) +
                                     l * (kv_dim / block_size) + k;

            float kmax = 0.f;
            float vmax = 0.f;

            for (size_t m = 0; m < block_size; m++) {
              float kval = fabs((float)dequant_keys[quant_loc + k * block_size + m]);
              kmax       = fmax(kmax, kval);

              float vval = fabs((float)dequant_values[quant_loc + k * block_size + m]);
              vmax       = fmax(vmax, vval);
            }

            const float dk  = kmax / ((1 << 7) - 1);
            const float idk = dk ? 1.f / dk : 0.f;

            q8_keys_scales[key_loc] = dk;

            const float dv  = vmax / ((1 << 7) - 1);
            const float idv = dv ? 1.f / dv : 0.f;

            q8_values_scales[key_loc] = dv;

            for (size_t m = 0; m < block_size; m++) {
              q8_keys_quant[quant_loc + k * block_size + m] =
                  (int8_t)roundf(dequant_keys[quant_loc + k * block_size + m] * idk);

              q8_values_quant[quant_loc + k * block_size + m] =
                  (int8_t)roundf(dequant_values[quant_loc + k * block_size + m] * idv);
            }
          }
        }
      }
    }

    __DEBUG("kv-convert: storing converted KV to file");
    cpu_fs.write((char*)q8_keys_quant.data(), q8_keys_quant.size() * sizeof(int8_t));
    cpu_fs.write((char*)q8_values_quant.data(), q8_values_quant.size() * sizeof(int8_t));

    cpu_fs.write((char*)q8_keys_scales.data(), q8_keys_scales.size() * sizeof(float));
    cpu_fs.write((char*)q8_values_scales.data(), q8_values_scales.size() * sizeof(float));
  } else {
    __DEBUG("kv-convert: storing converted KV to file");
    cpu_fs.write((char*)dequant_keys.data(), dequant_keys.size() * sizeof(float));
    cpu_fs.write((char*)dequant_values.data(), dequant_values.size() * sizeof(float));
  }

  cpu_fs.flush();
  cpu_fs.close();

  __DEBUG("kv-convert: done converting {} to {} in {} usec",
          nsp_cache_path.string(),
          cpu_cache_path.string(),
          start.elapsed_usec());

  return true;
}

// Registrator instance
static OnLoad regy([]() {
  Dialog::__register("kv-share",
                     [](std::shared_ptr<Env> env, const std::string& name, const json& conf) {
                       return (Dialog*)new KvShareDialog(env, name, conf);
                     });
});

void needKvShareDialog() {}

}  // namespace qualla