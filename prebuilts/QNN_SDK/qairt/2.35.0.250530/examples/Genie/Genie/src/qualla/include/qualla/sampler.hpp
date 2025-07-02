//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_SAMPLER_HPP
#define QUALLA_SAMPLER_HPP

#include <memory>
#include <random>
#include <span>
#include <string>
#include <vector>

#include "qualla/context.hpp"
#include "qualla/detail/exports.h"
#include "qualla/detail/json.hpp"
#include "qualla/detail/tensor.hpp"

namespace qualla {
typedef std::function<void(const uint32_t, const void*, const uint32_t, int32_t*)>
    SamplerCbFunction;
typedef std::function<void(const uint32_t, const void*, const uint32_t, int32_t*, const void*)>
    SamplerUserDataCbFunction;
class Sampler : public State {
 public:
  QUALLA_API Sampler(Context& ctx, const std::string& type, const qualla::json& conf);
  QUALLA_API ~Sampler();

  // Sample a single token from logits
  QUALLA_API int32_t process(Tensor& logits);

  // Sample a single token and output probabilities
  // Probs are appended to the existing vector
  QUALLA_API int32_t process(Tensor& logits, std::vector<float>& probs, bool out_tok = true);

  QUALLA_API std::vector<int32_t> process(Tensor& logits,
                                          std::vector<float>& probs,
                                          int32_t num_return);

  QUALLA_API bool save(const std::string& name);
  QUALLA_API bool restore(const std::string& name);
  QUALLA_API void reset();
  QUALLA_API void applyConfig(const qualla::json& conf);

  // Get sampler type
  const std::string& type() const { return _type; }

  // Get sampler role
  const std::string& role() const { return _role; }

  // Get sampler params
  bool greedy() const { return _greedy; }
  bool gumbel() const { return _gumbel; }
  int32_t seed() const { return _seed; }

  // Get reference to the random number generator
  std::mt19937& rng() { return _rng; }

  // Get sampler params
  float temp() const { return _temp; }
  size_t top_k() const { return _top_k; }
  float top_p() const { return _top_p; }

  // Set sampler params
  void temp(float t) { _temp = t; }
  void top_k(size_t k) { _top_k = k; }
  void top_p(float p) { _top_p = p; }

  // Create Sampler instance
  QUALLA_API static std::unique_ptr<Sampler> create(Context& ctx, std::istream& json_stream);
  QUALLA_API static std::unique_ptr<Sampler> create(Context& ctx, const std::string& json_str);
  QUALLA_API static std::unique_ptr<Sampler> create(Context& ctx, const qualla::json& conf = {});

  QUALLA_API static void registerProcessCallBack(std::string name,
                                                 qualla::SamplerCbFunction callback);
  QUALLA_API static void registerUserDataCallBack(std::string name,
                                                  qualla::SamplerUserDataCbFunction callback,
                                                  const void* userData);

  QUALLA_API std::vector<int32_t> process(Tensor& logits,
                                          std::vector<float>& probs,
                                          int32_t num_return,
                                          size_t topn_probs    = 0,
                                          int32_t n_vocab_trim = -1);

 protected:
  std::string _type;  // sampler type
  std::string _role;  // sampler role (primary, secondary, ...)
  Context& _ctx;      // reference to the context
  Env& _env;          // reference to the environment
  static std::unordered_map<std::string, std::tuple<qualla::SamplerCbFunction, qualla::SamplerUserDataCbFunction, const void*>> _samplerCbFunctionMap;
  std::mt19937 _rng;
  int32_t _seed{-1};
  bool _greedy{false};
  bool _gumbel{false};
  float _temp{0.1};
  size_t _top_k{0};
  float _top_p{0.8};
  std::string _customProcessCallbackName;
  template <typename T>
  int32_t basic_process(Tensor& logits, std::vector<float>* probs_out, bool samp_tok);
  template <typename T>
  std::vector<int32_t> basic_process(Tensor& logits, std::vector<float>& probs, int32_t num_return);
  template <typename T>
  std::vector<int32_t> custom_process(Tensor& logits, int numTokens);
  template <typename T>
  std::vector<int32_t> basic_process(Tensor& logits,
                                     std::vector<float>& probs,
                                     int32_t num_return,
                                     size_t topn_probs,
                                     int32_t n_vocab_trim);
};

}  // namespace qualla

#endif  // QUALLA_SAMPLER_HPP
