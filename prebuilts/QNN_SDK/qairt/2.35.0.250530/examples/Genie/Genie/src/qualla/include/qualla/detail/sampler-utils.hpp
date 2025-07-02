//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_DETAIL_SAMPLER_UTILS_HPP
#define QUALLA_DETAIL_SAMPLER_UTILS_HPP

#ifdef _MSC_VER
#pragma warning(disable : 4068)
#endif

#include <functional>
#include <random>
#include <span>
#include <string>

#include "qualla/detail/preproc.hpp"
#include "qualla/detail/tensor.hpp"

#define TOPP_SAMPLER_INITIAL_PARTITION_POINT 4096

namespace qualla {

typedef std::mt19937 rng_t;

// Various sampling utilities.

static double sampleFromUniform(rng_t& rng) {
  int a         = rng() >> 5;
  int b         = rng() >> 6;
  double sample = (a * 67108864.0 + b) / 9007199254740992.0;
  return sample;
}

static inline double sampleFromGumbel(rng_t& rng) {
  double tiny    = 1.1754943508222875e-38;
  double eps     = 1.1920928955078125e-07;
  double uniform = sampleFromUniform(rng);
  double gumbel  = -std::log(-std::log(tiny + uniform * (1. - eps - tiny)));
  return gumbel;
}

// Returns the index of an element chosen by applying the given probability distribution.
template <typename T>
static int32_t sampleFromProbs(const std::span<T> probs, rng_t& rng) {
  static_assert(std::is_floating_point<T>::value);
  std::discrete_distribution<> dist(probs.begin(), probs.end());
  return dist(rng);
}

// Returns the index of the element chosen by the Gumbel max algorithm
template <typename T>
static int32_t sampleUsingGumbelMax(const std::span<T> log_probs, rng_t& rng) {
  static_assert(std::is_floating_point<T>::value);
  int32_t max_purturbed_logit = std::numeric_limits<int32_t>::min();
  int32_t max_idx             = 0;

  for (int32_t i = 0; i < log_probs.size(); i++) {
    float purturbed_logit = log_probs[i] + sampleFromGumbel(rng);
    if (purturbed_logit > max_purturbed_logit) {
      max_purturbed_logit = purturbed_logit;
      max_idx             = i;
    }
  }
  return max_idx;
}

// Add gumbel noise to a set of logits
template <typename T>
void addGumbelNoise(std::vector<T>& log_probs, rng_t& rng) {
  static_assert(std::is_floating_point<T>::value);
  for (int32_t i = 0; i < log_probs.size(); i++) {
    log_probs[i] = log_probs[i] + sampleFromGumbel(rng);
  }
}

// Returns the index of the top token.
template <typename T>
static int32_t argmax(const std::span<T> probs) {
  static_assert(std::is_floating_point<T>::value);
  auto result = std::max_element(probs.begin(), probs.end());
  size_t id   = std::distance(probs.begin(), result);

  return int32_t(id);
}

// Get top-p elements from the input vector (vec)
// note:
//  -  element in the vec is a pair of index and probability
//  -  vec will be inplace modified
//  -  finally the first n_remain elements in vec will be the top-p elements, but UNSORTED
//  -  n_remain elements will be returned
//  -  first_try_pos can be used to speedup processing heuristically,
//     set to -1 to disable heuristical speedup
template <typename T>
size_t partitionTopP(std::vector<std::pair<int32_t, T>>& vec,
                     float top_p,
                     int32_t first_try_pos,
                     int32_t min_keep = 1) {
  static_assert(std::is_floating_point<T>::value);
  size_t i_start = 0;
  size_t i_end   = vec.size();  // not included

  T accum_left_sum = 0.0;  // sum of vec[:i_start]  (i_start is not included)
  T subarray_sum   = 1;    // sum of vec[i_start:i_end]  (i_end is not included)
  auto greater     = [](const std::pair<int32_t, T>& a, const std::pair<int32_t, T>& b) {
    return a.second > b.second;
  };

  size_t size                  = i_end - i_start;
  size_t m                     = size >> 1;
  size_t closest_partition_pos = vec.size();  // the closest partition position to min_keep,
                                              // but larger than min_keep

  if (first_try_pos > 0) {
    m = std::min((size_t)(first_try_pos), size);
  }

  while (i_start < i_end) {
    if (m == 0) {
      break;
    }

    if (i_start + m > min_keep) {
      closest_partition_pos = std::min(closest_partition_pos, i_start + m);
    }

    // partition vec[i_start:i_end] at the position "i_start+m"
    // [* * * * * * * * * * * * * * * * * * * * * * * * *]
    //      ^                 ^                     ^
    //     i_start        i_start + m             i_end (not included)

    // which satisfies:
    //    - any element in vec[i_start:i_start+m] is larger  than vec[i_start+m]
    //    - any element in vec[i_start+m+1:i_end] is smaller than vec[i_start+m]
    std::nth_element(
        vec.begin() + i_start, vec.begin() + i_start + m, vec.begin() + i_end, greater);

    // calculate the sum of vec[i_start:i_start + m]  (i_start + m is not included)
    T subarray_left_sum = 0;
    for (size_t i = i_start; i < i_start + m; i++) {
      subarray_left_sum += vec[i].second;
    }

    if (subarray_left_sum + accum_left_sum < top_p) {
      // do next iter on right sub array [i_start+m:i_end]
      i_start = i_start + m;
      m       = (i_end - i_start) >> 1;
      accum_left_sum += subarray_left_sum;
      subarray_sum = subarray_sum - subarray_left_sum;
    } else {
      // do next iter on left sub array [i_start:i_start+m]
      i_end = i_start + m;
      m     = (i_end - i_start) >> 1;
    }
  }

  size_t n_remain = i_start + 1;
  if (n_remain < min_keep) {
    std::nth_element(
        vec.begin() + n_remain, vec.begin() + min_keep, vec.begin() + closest_partition_pos);
    n_remain = min_keep;
  }
  return n_remain;
}

template <typename T>
struct IndexedQuantLogits {
  std::mt19937& rng;
  Tensor logitsTensor;
  std::span<const T> logits;
  std::vector<float> probs;
  std::vector<int32_t> indices;
  bool probs_valid;
  bool sorted;

  IndexedQuantLogits(Tensor logitsTensor, std::mt19937& r)
      : rng(r),
        logitsTensor(logitsTensor),
        probs(logitsTensor.getSize(), 0.f),
        indices(logitsTensor.getSize()),
        probs_valid(false),
        sorted(false) {
    std::iota(indices.begin(), indices.end(), 0);
    logits = std::span(reinterpret_cast<T*>(logitsTensor.getData()), logitsTensor.getSize());
  }

  size_t size(void) const { return logits.size(); }

  // Performs a partial sort or a full sort depending on k.
  size_t sort(size_t k = 0) {
    size_t logits_size = logits.size();

    k = k == 0 ? logits_size : k;
    k = std::min(k, logits_size);

    std::partial_sort(
        indices.begin(), indices.begin() + k, indices.end(), [this](int32_t a, int32_t b) {
          return logits[a] > logits[b];
        });

    // FIXME: avoid overwriting input logits

    if (probs_valid) {
      std::vector<T> tmp(k);
      std::vector<float> tmpf(k);
      for (int32_t i = 0; i < k; i++) {
        tmp[i]  = logits[indices[i]];
        tmpf[i] = probs[indices[i]];
      }
      memcpy(const_cast<T*>(logits.data()), tmp.data(), k * sizeof(T));
      memcpy(probs.data(), tmpf.data(), k * sizeof(float));
    } else {
      std::vector<T> tmp(k);
      for (int32_t i = 0; i < k; i++) {
        tmp[i] = logits[indices[i]];
      }
      memcpy(const_cast<T*>(logits.data()), tmp.data(), k * sizeof(T));
    }

    sorted = true;
    return k;
  }

  void softmax_topk(float temp = 1.f, size_t k = 0, size_t n = 0) {
    QUALLA_ASSERT(temp > 0.f);
    QUALLA_ASSERT(k >= 0);
    QUALLA_ASSERT(k <= n);

    size_t logits_size = logits.size();
    k                  = k == 0 ? logits_size : k;
    // k *= 8;
    k      = std::min(k, logits_size);
    n      = this->sort(n);
    logits = logits.subspan(0, n);
    indices.resize(n);
    probs.resize(n);
    T max_logit;
    if (sorted) {
      max_logit = logits[0];
    } else {
      auto max_iter = std::max_element(logits.begin(), logits.end());
      max_logit     = *max_iter;
    }

    TensorQuantizationParams qp = logitsTensor.getQuantizationParams();
    auto scale                  = qp.scale;
    auto offset                 = qp.offset;
    float max_logit_float       = ((float)max_logit + offset) * scale;

    float max_scaled = max_logit_float / temp;
    float sum_exp    = 0.0f;

    auto multFactor  = scale / temp;
    auto additionVal = ((scale * offset) / temp) - max_scaled;

#pragma clang loop vectorize(enable)
    for (size_t i = 0; i < logits.size(); i++) {
      float p  = std::exp(((float)logits[i] * multFactor) + additionVal);
      probs[i] = p;
      sum_exp += p;
    }

#pragma clang loop vectorize(enable)
    for (size_t i = 0; i < logits.size(); i++) {
      probs[i] /= sum_exp;
    }

    probs_valid = true;
  }

  // Does softmax in-place given a set of logits and a scaling temperature.
  void softmax(float temp = 1.f) {
    QUALLA_ASSERT(temp > 0.f);

    T max_logit;

    if (sorted) {
      max_logit = logits[0];
    } else {
      auto max_iter = std::max_element(logits.begin(), logits.end());
      max_logit     = *max_iter;
    }

    TensorQuantizationParams qp = logitsTensor.getQuantizationParams();
    auto scale                  = qp.scale;
    auto offset                 = qp.offset;
    float max_logit_float       = ((float)max_logit + offset) * scale;

    float max_scaled = max_logit_float / temp;
    float sum_exp    = 0.0f;

    auto multFactor  = scale / temp;
    auto additionVal = ((scale * offset) / temp) - max_scaled;

#pragma clang loop vectorize(enable)
    for (size_t i = 0; i < logits.size(); i++) {
      float p  = std::exp(((float)logits[i] * multFactor) + additionVal);
      probs[i] = p;
      sum_exp += p;
    }

#pragma clang loop vectorize(enable)
    for (size_t i = 0; i < logits.size(); i++) {
      probs[i] /= sum_exp;
    }

    probs_valid = true;
  }

  void logSoftmax(float temp = 1.f) {
    QUALLA_ASSERT(temp > 0.f);
    float max_logit;

    if (sorted) {
      max_logit = logits[0];
    } else {
      auto max_iter = std::max_element(logits.begin(), logits.end());
      max_logit     = *max_iter;
    }

    TensorQuantizationParams qp = logitsTensor.getQuantizationParams();
    auto scale                  = qp.scale;
    auto offset                 = qp.offset;
    float max_logit_float       = ((float)max_logit + offset) * scale;

    // log(e^x / sum(e^x)) -> log(e^x) - log(sum(e^x))
    // We're still using the probs vector, despite the outputs technically
    // being log probabilities.

    float max_scaled = max_logit_float / temp;
    float sum_exp    = 0.0f;

    auto multFactor  = scale / temp;
    auto additionVal = ((scale * offset) / temp) - max_scaled;

#pragma clang loop vectorize(enable)
    for (size_t i = 0; i < logits.size(); i++) {
      float p  = ((float)logits[i] * multFactor) + additionVal;
      probs[i] = p;
      sum_exp += std::exp(p);
    }

    float log_sum_exp = std::log(sum_exp);
#pragma clang loop vectorize(enable)
    for (size_t i = 0; i < logits.size(); i++) {
      probs[i] -= log_sum_exp;
    }

    probs_valid = true;
  }

  // Performs top-k
  void topK(int32_t k) {
    QUALLA_ASSERT(k > 0);
    k = this->sort(k);

    logits = logits.subspan(0, k);
    probs.resize(k);
    indices.resize(k);
  }

  // Performs top-p in-place.
  // Note: the remained logits/probs are UNSORTED
  void topP(float p, int32_t min_keep = 1) {
    if (p >= 1) return;

    if (!probs_valid) this->softmax();

    if (sorted) {
      // The probs are sorted, so find top-p elements directly

      // Compute the cumulative probabilities
      float cum_sum    = 0.0;
      size_t last_idx  = logits.size() - 1;
      size_t n_to_trim = 0;

      for (size_t i = last_idx; i > 0; --i) {
        cum_sum += probs[i];
        if (cum_sum <= 1.0 - p) {
          n_to_trim++;
        } else {
          break;
        }
      }

      size_t n_remain = logits.size() - n_to_trim;
      if (n_remain < min_keep) {
        n_remain += min_keep - n_remain;
      }

      logits = logits.first(n_remain);
      probs.resize(n_remain);
      indices.resize(n_remain);

    } else {
      // The probs are not sorted, so using binary partition to find top-p elements,
      // which is much faster than sorting for large vocab size

      // pack index/logit into one array, to improve data locality
      std::vector<std::pair<int32_t, float>> elements;
      elements.reserve(logits.size());

#pragma clang loop vectorize(enable)
      for (size_t i = 0; i < logits.size(); ++i) {
        elements.emplace_back(indices[i], probs[i]);
      }

      // normally probs are only concentrated in 1-100 labels
      // so try the first parition with a not very large value (like 4096)
      // will speedup the topp_by_nth in most of the cases
      int first_try_pos = TOPP_SAMPLER_INITIAL_PARTITION_POINT;

      // however, if the logits size is small, we don't need to this heuristic acceleration
      if (logits.size() < first_try_pos * 2) first_try_pos = -1;
      size_t n_remain = partitionTopP(elements, p, first_try_pos, min_keep);

      indices.resize(n_remain);
      probs.resize(n_remain);
      std::vector<T> temp_logits(n_remain);

#pragma clang loop vectorize(enable)
      for (size_t i = 0; i < n_remain; i++) {
        indices[i]     = elements[i].first;
        probs[i]       = elements[i].second;
        temp_logits[i] = logits[elements[i].first];
      }
      memcpy(const_cast<T*>(logits.data()), temp_logits.data(), n_remain * sizeof(T));
      logits = logits.first(n_remain);
    }

    // The probabilities no longer add up to 1.
    probs_valid = false;
  }

  // Greedy sampling
  int32_t sampleGreedyUnsorted() {
    auto result = std::max_element(logits.begin(), logits.end());
    size_t id   = std::distance(logits.begin(), result);
    std::fill_n(probs.begin(), probs.size(), (float)0);
    probs[id]   = 1.0;
    probs_valid = true;
    return int32_t(id);
  }

  // Sampling from prob distribution
  int32_t sampleFromProbs() {
    QUALLA_ASSERT(probs_valid);
    int32_t idx = qualla::sampleFromProbs<float>(std::span{probs.data(), probs.size()}, rng);
    return int32_t(indices[idx]);
  }

  // Sampling with Gumbel Max
  int32_t sampleUsingGumbelMax() {
    QUALLA_ASSERT(probs_valid);
    // probs here must be log-probabilities
    int32_t idx = qualla::sampleUsingGumbelMax<float>(std::span{probs.data(), probs.size()}, rng);
    return int32_t(indices[idx]);
  }

  // add gumbel noise to the logits
  bool addGumbelNoise() {
    // probs here must be log-probabilities
    qualla::addGumbelNoise<float>(probs, rng);
    return true;
  }
};

}  // namespace qualla

#endif  // QUALLA_DETAIL_SAMPLER_UTILS_HPP
