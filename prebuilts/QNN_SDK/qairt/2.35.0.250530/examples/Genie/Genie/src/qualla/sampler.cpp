//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <ctime>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <random>
#include <span>
#include <string>
#include <unordered_map>

#include "qualla/detail/config.hpp"
#include "qualla/detail/onload.hpp"
#include "qualla/detail/sampler-utils.hpp"
#include "qualla/sampler.hpp"

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

namespace fs = std::filesystem;

namespace qualla {

Sampler::Sampler(Context& ctx, const std::string& type, const qualla::json& conf)
    : _type(type), _ctx(ctx), _env(ctx.env()) {
  __DEBUG("sampler-new: {} ctx {} config {}", type, ctx.name(), conf.dump());

  // Parse config
  using qc = qualla::Config;

  _role   = qc::optional<std::string>(conf, "role", "primary");
  _seed   = qc::optional<int32_t>(conf, "seed", -1);
  _greedy = qc::optional<bool>(conf, "greedy", _greedy);

  _gumbel = qc::optional(conf, "use-gumbel", false);
  _gumbel = qc::optional(conf, "gumbel", _gumbel);

  if (_type == "basic") {
    _temp   = qc::optional<float>(conf, "temp", 0.1);
    _top_k  = qc::optional<size_t>(conf, "top-k", 0);
    _top_p  = qc::optional<float>(conf, "top-p", 0.8);
    _greedy = (_temp <= 0.f || _top_k == 1);
    _rng.seed(_seed != -1 ? _seed : std::time(nullptr));
  } else if (_type == "custom") {
    _greedy                    = true;  // only support greedy sampling in custom sampler
    _customProcessCallbackName = qc::mandatory<std::string>(conf, "callback-name");
    if (_samplerCbFunctionMap.find(_customProcessCallbackName) == _samplerCbFunctionMap.end()) {
      __ERROR("callback-name {} passed not registered ", _customProcessCallbackName);
    }
  } else {
    __ERROR("Invalid sampler type ", _type);
  }
}

Sampler::~Sampler() {}

bool Sampler::restore(const std::string& name) {
  if (_type == "basic") {
    fs::path restore_path = std::filesystem::path(name) / fmt::format("sampler.{}.rng", _role);

    std::fstream f(restore_path, std::ios::in);
    if (!f.is_open()) {
      __ERROR("basic-sampler: failed to open {} for reading", restore_path.string());
      return false;
    }

    f >> _rng;
    f.close();

    return true;
  }
  __WARN("{}-sampler does not support restore", _type);
  return false;
}

bool Sampler::save(const std::string& name) {
  if (_type == "basic") {
    fs::path save_path = std::filesystem::path(name) / fmt::format("sampler.{}.rng", _role);

    std::fstream f(save_path, std::ios::out | std::ios::trunc);
    if (!f.is_open()) {
      __ERROR("basic-sampler: failed to open {} for writing", save_path.string());
      return false;
    }

    f << _rng;
    f.close();

    return true;
  }
  __WARN("{}-sampler does not support save", _type);
  return false;
}

void Sampler::reset() {
  if (_type == "basic") {
    // Just need to reinit rng
    _rng.seed(_seed);
  } else {
    __WARN("{}-sampler does not support reset", _type);
  }
}

int32_t Sampler::process(Tensor& logits) {
  if (_type == "basic") {
    // basic sampler bool fn
    switch (logits.getDataType()) {
      case TENSOR_DATATYPE_UFIXED_POINT_8: {
        return basic_process<uint8_t>(logits, nullptr, true);
      }
      case TENSOR_DATATYPE_UFIXED_POINT_16: {
        return basic_process<uint16_t>(logits, nullptr, true);
      }
      case TENSOR_DATATYPE_FLOAT_POINT_16: {
        return basic_process<uint16_t>(logits, nullptr, true);
      }
      case TENSOR_DATATYPE_FLOAT_32: {
        return basic_process<float>(logits, nullptr, true);
      }
      default: {
        __WARN("Unsupported datatype");
      }
    }
  } else if (_type == "custom") {
    // custom sampling fn
    std::vector<int32_t> last_tok_vec;
    switch (logits.getDataType()) {
      case TENSOR_DATATYPE_UFIXED_POINT_8: {
        last_tok_vec = custom_process<uint8_t>(logits, 1);
        break;
      }
      case TENSOR_DATATYPE_UFIXED_POINT_16: {
        last_tok_vec = custom_process<uint16_t>(logits, 1);
        break;
      }
      case TENSOR_DATATYPE_FLOAT_POINT_16: {
        last_tok_vec = custom_process<uint16_t>(logits, 1);
        break;
      }
      case TENSOR_DATATYPE_FLOAT_32: {
        last_tok_vec = custom_process<float>(logits, 1);
        break;
      }
      default: {
        __WARN("Unsupported datatype");
        return -1;
      }
    }
    return last_tok_vec[0];
  }
  return -1;
}

int32_t Sampler::process(Tensor& logits, std::vector<float>& probs, bool out_tok) {
  if (_type == "basic") {
    // basic sampler bool fn
    switch (logits.getDataType()) {
      case TENSOR_DATATYPE_UFIXED_POINT_8: {
        return basic_process<uint8_t>(logits, &probs, out_tok);
      }
      case TENSOR_DATATYPE_UFIXED_POINT_16: {
        return basic_process<uint16_t>(logits, &probs, out_tok);
      }
      case TENSOR_DATATYPE_FLOAT_POINT_16: {
        return basic_process<uint16_t>(logits, &probs, out_tok);
      }
      case TENSOR_DATATYPE_FLOAT_32: {
        return basic_process<float>(logits, &probs, out_tok);
      }
      default: {
        __WARN("Unsupported datatype");
      }
    }
  } else if (_type == "custom") {
    std::vector<int32_t> last_tok_vec;
    switch (logits.getDataType()) {
      case TENSOR_DATATYPE_UFIXED_POINT_8: {
        last_tok_vec = custom_process<uint8_t>(logits, 1);
        break;
      }
      case TENSOR_DATATYPE_UFIXED_POINT_16: {
        last_tok_vec = custom_process<uint16_t>(logits, 1);
        break;
      }
      case TENSOR_DATATYPE_FLOAT_POINT_16: {
        last_tok_vec = custom_process<uint16_t>(logits, 1);
        break;
      }
      case TENSOR_DATATYPE_FLOAT_32: {
        last_tok_vec = custom_process<float>(logits, 1);
        break;
      }
      default: {
        __WARN("Unsupported datatype");
        return -1;
      }
    }
    return last_tok_vec[0];
  }
  return -1;
}

std::vector<int32_t> Sampler::process(Tensor& logits,
                                      std::vector<float>& probs,
                                      int32_t num_return) {
  if (_type == "basic") {
    // basic sampler bool fn
    switch (logits.getDataType()) {
      case TENSOR_DATATYPE_UFIXED_POINT_8: {
        return basic_process<uint8_t>(logits, probs, num_return);
      }
      case TENSOR_DATATYPE_UFIXED_POINT_16: {
        return basic_process<uint16_t>(logits, probs, num_return);
      }
      case TENSOR_DATATYPE_FLOAT_POINT_16: {
        return basic_process<uint16_t>(logits, probs, num_return);
      }
      case TENSOR_DATATYPE_FLOAT_32: {
        return basic_process<float>(logits, probs, num_return);
      }
      default: {
        __WARN("Unsupported datatype");
      }
    }
  } else if (_type == "custom") {
    // custom sampling fn
    switch (logits.getDataType()) {
      case TENSOR_DATATYPE_UFIXED_POINT_8: {
        return custom_process<uint8_t>(logits, num_return);
      }
      case TENSOR_DATATYPE_UFIXED_POINT_16: {
        return custom_process<uint16_t>(logits, num_return);
      }
      case TENSOR_DATATYPE_FLOAT_POINT_16: {
        return custom_process<uint16_t>(logits, num_return);
      }
      case TENSOR_DATATYPE_FLOAT_32: {
        return custom_process<float>(logits, num_return);
      }
      default: {
        __WARN("Unsupported datatype");
      }
    }
  }
  return {-1};
}

template <typename T>
int32_t Sampler::basic_process(Tensor& logits, std::vector<float>* probs_out, bool tok_out) {
  const size_t n_vocab = _ctx.n_vocab();

  assert(logits.getSize() % n_vocab == 0);
  assert(logits.getSize() / n_vocab == 1);

  const float temp    = _temp;
  const int32_t top_k = _top_k;
  const float top_p   = _top_p;

  std::span<const T> logitsSpan =
      std::span(reinterpret_cast<T*>(logits.getData()), logits.getSize());
  __DEBUG("input-logits: {} ... {}", logitsSpan.first(10), logitsSpan.last(10));

  IndexedQuantLogits<T> indexed_logits(logits, _rng);

  int32_t id = -1;

  if (_greedy) {
    // Greedy sampling
    id = indexed_logits.sampleGreedyUnsorted();
  } else {
    // Temperature sampling
    if (top_k > 0) {
      indexed_logits.topK(top_k);
    }

    indexed_logits.topP(top_p, 1);

    if (_gumbel) {
      indexed_logits.logSoftmax(temp);
      id = tok_out ? indexed_logits.sampleUsingGumbelMax() : -1;
    } else {
      indexed_logits.softmax(temp);
      id = tok_out ? indexed_logits.sampleFromProbs() : -1;
    }
  }

  // Output probability distribution
  if (probs_out) {
    QUALLA_ASSERT(indexed_logits.probs_valid);

    // Expand the output vector and fill it with the default values
    probs_out->resize(probs_out->size() + n_vocab,
                      _gumbel ? -std::numeric_limits<float>::infinity() : 0);

    auto p = std::span(probs_out->data(), probs_out->size()).last(n_vocab);
    for (size_t i = 0; i < indexed_logits.size(); i++) {
      int t = (int)indexed_logits.indices[i];
      p[t]  = indexed_logits.probs[i];
    }
  }

  return id;
}

// return multiple tokens - top_k after processing, temperature, top_p, gumbel, etc
template <typename T>
std::vector<int32_t> Sampler::basic_process(Tensor& logits,
                                            std::vector<float>& probs,
                                            int32_t num_return) {
  const size_t n_vocab = _ctx.n_vocab();

  assert(logits.getSize() % n_vocab == 0);
  assert(logits.getSize() / n_vocab == 1);

  const float temp  = _temp;
  const float top_p = _top_p;
  num_return        = num_return <= 0 ? n_vocab : num_return;

  std::span<const T> logitsSpan =
      std::span(reinterpret_cast<T*>(logits.getData()), logits.getSize());
  __DEBUG("input-logits: {} ... {}", logitsSpan.first(10), logitsSpan.last(10));

  IndexedQuantLogits<T> indexed_logits(logits, _rng);

  std::vector<int32_t> ids;

  // Temperature sampling
  indexed_logits.topP(top_p, 1);
  // add gumbel noise to the logits
  if (_gumbel) {
    indexed_logits.logSoftmax(temp);
    indexed_logits.addGumbelNoise();
  } else {
    indexed_logits.softmax(temp);
  }

  num_return =
      num_return <= indexed_logits.indices.size() ? num_return : indexed_logits.indices.size();
  indexed_logits.topK(num_return);
  ids = indexed_logits.indices;
  for (int i = 0; i < indexed_logits.probs.size(); i++) {
    probs[i] = indexed_logits.probs[i];
  }

  return ids;
}

std::vector<int32_t> Sampler::process(Tensor& logits,
                                      std::vector<float>& probs,
                                      int32_t num_return,
                                      size_t topn_probs,
                                      int32_t n_vocab_trim) {
  if (_type == "basic") {
    // basic sampler bool fn
    switch (logits.getDataType()) {
      case TENSOR_DATATYPE_UFIXED_POINT_8: {
        return basic_process<uint8_t>(logits, probs, num_return, topn_probs, n_vocab_trim);
      }
      case TENSOR_DATATYPE_UFIXED_POINT_16: {
        return basic_process<uint16_t>(logits, probs, num_return, topn_probs, n_vocab_trim);
      }
      case TENSOR_DATATYPE_FLOAT_POINT_16: {
        return basic_process<uint16_t>(logits, probs, num_return, topn_probs, n_vocab_trim);
      }
      case TENSOR_DATATYPE_FLOAT_32: {
        return basic_process<float>(logits, probs, num_return, topn_probs, n_vocab_trim);
      }
      default: {
        __WARN("Unsupported datatype");
        return {};
      }
    }
  }
  return {};
}

template <typename T>
std::vector<int32_t> Sampler::basic_process(Tensor& logits,
                                            std::vector<float>& probs,
                                            int32_t num_return,
                                            size_t topn_probs,
                                            int32_t n_vocab_trim) {
  const size_t n_vocab = n_vocab_trim > 0 ? n_vocab_trim : _ctx.n_vocab();

  assert(logits.getSize() % n_vocab == 0);
  assert(logits.getSize() / n_vocab == 1);

  const float temp  = _temp;
  const float top_p = _top_p;
  num_return        = num_return <= 0 ? n_vocab : num_return;

  std::span<const T> logitsSpan =
      std::span(reinterpret_cast<T*>(logits.getData()), logits.getSize());
  __DEBUG("input-logits: {} ... {}", logitsSpan.first(10), logitsSpan.last(10));

  IndexedQuantLogits<uint16_t> indexed_logits(logits, _rng);

  std::vector<int32_t> ids;

  // Temperature sampling
  indexed_logits.topP(top_p, 1);

  indexed_logits.softmax_topk(temp, num_return, topn_probs);

  num_return =
      num_return <= indexed_logits.indices.size() ? num_return : indexed_logits.indices.size();
  // indexed_logits.topK(num_return);

  indexed_logits.logits = indexed_logits.logits.first(num_return);
#pragma clang loop vectorize(enable)
  for (size_t i = 0; i < num_return; i++) {
    probs[i] = indexed_logits.probs[i];
    ids.push_back(indexed_logits.indices[i]);
  }
  return ids;
}

template <typename T>
std::vector<int32_t> Sampler::custom_process(Tensor& logits, int numTokens) {
  auto retToken = std::vector<int32_t>(numTokens);
  std::vector<float> logitVector;
  logitVector.reserve(logits.getSize());
  std::span<const T> logitsSpan =
      std::span(reinterpret_cast<T*>(logits.getData()), logits.getSize());
  TensorQuantizationParams qp = logits.getQuantizationParams();
  auto scale                  = qp.scale;
  auto offset                 = qp.offset;
  for (int i = 0; i < logits.getSize(); i++) {
    logitVector[i] = ((float)logitsSpan[i] + offset) * scale;
  }
  if (std::get<0>(_samplerCbFunctionMap[_customProcessCallbackName]) == nullptr) {
    auto userData = std::get<2>(_samplerCbFunctionMap[_customProcessCallbackName]);
    std::get<1>(_samplerCbFunctionMap[_customProcessCallbackName])(
        (logits.getSize() * sizeof(float)),
        logitVector.data(),
        numTokens,
        retToken.data(),
        userData);
  } else {
    std::get<0>(_samplerCbFunctionMap[_customProcessCallbackName])(
        (logits.getSize() * sizeof(float)), logitVector.data(), numTokens, retToken.data());
  }
  return retToken;
}

void Sampler::applyConfig(const qualla::json& conf) {
  if (conf.contains("type")) {
    _type = conf["type"];
  }
  if (_type == "basic") {
    if (conf.contains("seed")) _seed = conf["seed"];
    if (conf.contains("temp")) _temp = conf["temp"];

    if (conf.contains("top-k")) _top_k = conf["top-k"];
    if (conf.contains("top-p")) _top_p = conf["top-p"];
  } else if (_type == "custom") {
    if (conf.contains("callback-name")) {
      _customProcessCallbackName = conf["callback-name"];
      if (_samplerCbFunctionMap.find(_customProcessCallbackName) == _samplerCbFunctionMap.end()) {
        __ERROR("callback-name {} passed not registered ", _customProcessCallbackName);
      }
    }
  } else {
    __ERROR("Invalid sampler type ", _type);
  }
}

std::unordered_map<
    std::string,
    std::tuple<qualla::SamplerCbFunction, qualla::SamplerUserDataCbFunction, const void*>>
    Sampler::_samplerCbFunctionMap;
void Sampler::registerProcessCallBack(std::string name, qualla::SamplerCbFunction callback) {
  _samplerCbFunctionMap[name] = std::make_tuple(callback, nullptr, nullptr);
}

void Sampler::registerUserDataCallBack(std::string name,
                                       qualla::SamplerUserDataCbFunction callback,
                                       const void* userData) {
  _samplerCbFunctionMap[name] = std::make_tuple(nullptr, callback, userData);
}

std::unique_ptr<Sampler> Sampler::create(Context& ctx, const qualla::json& conf) {
  using qc         = qualla::Config;
  std::string type = qc::optional<std::string>(conf, "type", "basic");

  return std::unique_ptr<Sampler>((Sampler*)new Sampler(ctx, type, conf));
}

std::unique_ptr<Sampler> Sampler::create(Context& ctx, std::istream& json_stream) {
  return create(ctx, json::parse(json_stream));
}

std::unique_ptr<Sampler> Sampler::create(Context& ctx, const std::string& json_str) {
  return create(ctx, json::parse(json_str));
}

}  // namespace qualla
