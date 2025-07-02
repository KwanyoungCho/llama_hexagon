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

class SpecDecDialog : public Dialog {
 public:
  SpecDecDialog(std::shared_ptr<Env> env, const std::string& name, const json& conf);

  virtual bool process(std::vector<int32_t>& tokens, Dialog::Callback callback) override;

  virtual bool process(std::vector<int32_t>& tokens, DialogCallback callback) override {
    return false;
  }

  virtual void reset() override;

 private:
  int32_t _draft_len;  // Number of draft tokens
  bool _parallel;      // Enable parallel processing (where possible)

  // For keeping track of the number of tokens that were accepted in each iteration.
  std::vector<int32_t> _accepted_counts;

  Sampler& _d_sampler;  // Draft sampler
  Sampler& _t_sampler;  // Target sampler

  // Token acceptor, called for each accepted token.
  // Returns true to continue, false to stop
  using Acceptor = std::function<bool(int32_t token)>;

  // Follow on processing of the tokens
  bool processFollowOnGeneration(std::vector<int32_t>& tokens,
                                 Tensor& t_logits,
                                 Tensor& d_logits,
                                 Dialog::Callback callback);

  // Rejection sampling.
  // Returns number of accepted tokens
  size_t rejectionSampling(std::span<int32_t> tokens,
                           Tensor& target_logits,
                           std::span<float> draft_probs,
                           Acceptor accept);

  int32_t sampleFromModifiedDist(std::span<float> src0_dst, std::span<float> src1);
};

SpecDecDialog::SpecDecDialog(std::shared_ptr<Env> env, const std::string& name, const json& conf)
    : Dialog(env, name, conf),
      _d_sampler(_sampler.contains("draft") ? *_sampler["draft"] : *_sampler["target"]),
      _t_sampler(*_sampler["target"]) {
  _draft_len = qc::optional<int32_t>(conf, "draft-len", 3);
  _parallel  = qc::optional<bool>(conf, "parallel", false);

  // Check all underlying components for correct types an config
  // If something is not right we set our error state that can be checked later

  if (!_sampler.contains("target")) {
    State::fatal("\"target\" sampler not present in config!");
    return;
  }

  if (!_engine.contains("target")) {
    State::fatal("\"target\" engine not present in config!");
    return;
  }
  if (!_engine.contains("draft")) {
    State::fatal("\"draft\" engine not present in config!");
    return;
  }

  _accepted_counts.resize(_draft_len + 1, 0);
}

int32_t SpecDecDialog::sampleFromModifiedDist(std::span<float> src0_dst, std::span<float> src1) {
  //  [max(prob_target[x] - prob_draft[x], 0.f) for all x in vocab]
  size_t size = src0_dst.size();

  if (_t_sampler.gumbel()) {
    // Avoid going in the denormal zone.
    float tiny = 1.1754943508222875e-38;

#pragma clang loop vectorize(enable) unroll_count(4)
    for (size_t i = 0U; i < size; i++) {
      float p_src0 = std::exp(src0_dst[i]);
      float p_src1 = std::exp(src1[i]);
      src0_dst[i]  = std::log(std::max(tiny, p_src0 - p_src1));
    }

    // NOTE: The output logps_target is unnormalized since we use Gumbel trick.
    //       If we use standard multinomial sampling, normalization should be added.

  } else {
    float sum = 0.0;  // Unlikely to overflow (?)
#pragma clang loop vectorize(enable) unroll_count(4)
    for (size_t i = 0U; i < size; i++) {
      float num = std::max(0.f, src0_dst[i] - src1[i]);
      sum += num;
      src0_dst[i] = num;
    }
    // Normalize
#pragma clang loop vectorize(enable) unroll_count(4)
    for (size_t i = 0U; i < size; i++) {
      src0_dst[i] /= sum;
    }
  }

  if (_t_sampler.greedy()) return argmax(src0_dst);

  if (_t_sampler.gumbel()) return sampleUsingGumbelMax(src0_dst, _t_sampler.rng());

  // Skipping softmax since the probs are already normalized
  return sampleFromProbs(src0_dst, _t_sampler.rng());
}

size_t SpecDecDialog::rejectionSampling(std::span<int32_t> tokens,
                                        Tensor& target_logits,
                                        std::span<float> draft_probs,
                                        Acceptor accept) {
  const size_t n_vocab = _ctx->n_vocab();
  const size_t n_tok   = tokens.size();

  assert(tokens.size() == draft_probs.size() / n_vocab);
  assert(target_logits.getSize() == draft_probs.size() + n_vocab);

  // Rejection sampling:
  // For each token in the n_tok tokens sampled from the draft model:
  // 1. Determine the probability of that token being accepted by the target model
  // 2. Accept the token with probability = prob_target[tok] / prob_draft[tok] (clamped to [0, 1])
  // 3. If the token is rejected, resample a new token from the following distribution:
  //      [max(prob_target[x] - prob_draft[x], 0.f) for all x in vocab]
  int32_t t_tok;
  size_t n_accepted = 0;

  std::vector<float> target_probs;

  for (int32_t i = 0; i < n_tok; i++) {
    int32_t d_tok = tokens[i];

    Tensor index_t_logits = target_logits.getIndexedTensor(i, n_vocab);

    if (_t_sampler.greedy()) {  // Invoked for Custom and basic with greedy sampling
      t_tok = _t_sampler.process(index_t_logits);
      if (t_tok != d_tok) {
        // Reject
        break;
      }
    } else {
      target_probs.clear();
      t_tok = _t_sampler.process(index_t_logits, target_probs, false);  // only probs, no token

      // Acceptance threshold
      double threshold;
      float prob_draft  = draft_probs[i * n_vocab + d_tok];
      float prob_target = target_probs[d_tok];

      if (_t_sampler.gumbel()) {
        threshold = std::exp(double(prob_target) - double(prob_draft));
      } else {
        threshold = double(prob_target) / double(prob_draft);
      }

      double r = sampleFromUniform(_t_sampler.rng());
      if (r > threshold) {
        // Reject
        break;
      }
    }
    // Accepted!
    ++n_accepted;
    if (!accept(d_tok)) return n_accepted;
  }

  // Sample an extra token either from the target distribution or the modified distribution
  if (n_accepted == n_tok) {
    Tensor t = target_logits.getIndexedTensor(n_tok, n_vocab, true);
    t_tok    = _t_sampler.process(t);
  } else if (!_t_sampler.greedy()) {
    // Resample from modified distribution.
    t_tok = sampleFromModifiedDist(std::span{target_probs.data(), target_probs.size()},
                                   draft_probs.subspan(n_accepted * n_vocab, n_vocab));
  }  // for greedy, t_tok should be already valid from the loop above

  ++n_accepted;
  accept(t_tok);

  return n_accepted;
}

bool SpecDecDialog::processFollowOnGeneration(std::vector<int32_t>& tokens,
                                              Tensor& t_logits,
                                              Tensor& d_logits,
                                              Dialog::Callback callback) {
  const size_t n_vocab = _ctx->n_vocab();

  bool keep_generating = true;

  // A buffer for tokens to be decoded (one at a time, per the Middleware's request)
  std::vector<int32_t> decode_buf(1, 0);

  // Decode new token.
  // Return true to continue generation, and false otherwise
  auto decode_token = [&](int32_t t) {
    decode_buf[0] = _last_tok = t;

    if (_ctx->is_eos(t)) {
      keep_generating = false;
      callback("", Sentence::END);
    } else {
      keep_generating = callback(_tokenizer->decode(decode_buf), Sentence::CONTINUE);
    }

    return keep_generating;
  };

  auto& t_engine = *_engine["target"];
  auto& d_engine = *_engine["draft"];

  // Buffers for all the tokens that need to be considered for each iteration
  std::vector<int32_t> toks_to_target(_draft_len + 1);
  std::vector<int32_t> toks_to_draft(2);

  // Buffer for all the probability distributions from the draft sampler
  std::vector<float> d_probs(n_vocab * _draft_len);

  toks_to_target.assign(1, _last_tok);
  toks_to_draft.assign(1, _last_tok);

  // Draft n_past, either in sync with n_past or one token behind (accepted-all)
  size_t d_n_past = _n_past;

  Timer start;

  while (!State::canceled() && keep_generating) {
    // Step 1: Use draft model to decode draft_len (aka gamma) tokens, and accumulate probabilities
    d_probs.clear();

    for (int32_t i = 0; i < _draft_len; i++) {
      if (d_n_past + toks_to_draft.size() > _ctx->size()) {
        __WARN(
            "Context limit exceeded ({} + {} > {})", d_n_past, toks_to_target.size(), _ctx->size());
        _kpis.generate.update(start.elapsed_usec());

        // Log latest KPIs in a single line
        __KPIS("{}", kpis().dump(" "));
        callback("", Sentence::END);
        return true;
      }

      if (!d_engine.process(toks_to_draft, d_logits))
        return Dialog::abort("draft engine gen processing failed", callback);

      d_n_past += toks_to_draft.size();

      if (!d_engine.updateKV(d_n_past))
        return Dialog::abort("draft context size exceeded", callback);

      int32_t token = _d_sampler.process(d_logits, d_probs);
      toks_to_draft.assign(1, token);
      toks_to_target.push_back(token);

      if (_ctx->is_eos(token)) break;
    }

    // Step 2: run the target model on the draft tokens
    if (_n_past + toks_to_target.size() > _ctx->size()) {
      __WARN("Context limit exceeded ({} + {} > {})", _n_past, toks_to_target.size(), _ctx->size());
      callback("", Sentence::END);
      _kpis.generate.update(start.elapsed_usec());

      // Log latest KPIs in a single line
      __KPIS("{}", kpis().dump(" "));
      return true;
    }

    std::vector<int32_t> attention_map(toks_to_target.size());
    std::iota(attention_map.begin(), attention_map.end(), -1);
    size_t n_tok_t =
        t_engine.process(toks_to_target, attention_map, t_logits, true /* all logits */);
    if (n_tok_t != toks_to_target.size())
      return Dialog::abort("target engine gen processing failed", callback);

    // Step 3: accept or reject draft tokens
    size_t n_accepted =
        rejectionSampling(std::span{toks_to_target.data(), toks_to_target.size()}.subspan(1),
                          t_logits,
                          std::span{d_probs.data(), d_probs.size()},
                          decode_token);

    _n_generated += n_accepted;
    _n_past += n_accepted;

    // Update stats
    _accepted_counts[n_accepted - 1]++;

    // Accepted all?
    if (n_accepted == _draft_len + 1) {
      // Grab the last 2 tokens
      toks_to_draft.assign({toks_to_target[_draft_len], _last_tok});
      d_n_past = _n_past - 1;
    } else {
      // Grab only the last token
      toks_to_draft.assign(1, _last_tok);
      d_n_past = _n_past;
    }

    toks_to_target.assign(1, _last_tok);

    __DEBUG("spec-dec: draft_len {} n_generated {} n_accepted {} n_past {}",
            _draft_len,
            _n_generated,
            n_accepted,
            _n_past);

    std::vector<bool> selected(attention_map.size(), false);
    selected[0]   = true;  // first token is selected always
    auto last_sel = 0;
    for (int i = n_accepted - 1; i != 0; i = attention_map[i]) {
      selected[i] = true;
      last_sel    = i > last_sel ? i : last_sel;
    }
    selected.resize(last_sel + 1);  // trim away rejected tokens

    // Step 4: commit accepted tokens to kv-caches
    if (!t_engine.updateKV(_n_past, selected))
      return Dialog::abort("target context size exceeded", callback);
    if (!d_engine.updateKV(d_n_past)) return Dialog::abort("draft context size exceeded", callback);
  }

  if (d_n_past != _n_past) {
    // The draft engine needs to process one last token to catch up
    toks_to_draft.resize(1);
    if (!d_engine.process(toks_to_draft))
      return Dialog::abort("draft engine gen processing failed", callback);
    if (!d_engine.updateKV(_n_past)) return Dialog::abort("draft context size exceeded", callback);
  }

  return true;
}
bool SpecDecDialog::process(std::vector<int32_t>& tokens, Dialog::Callback callback) {
  // Check for prev failures and bail out early
  if (State::failed()) return false;

  Timer start;

  // Vector for storing logits.
  // Allocated & filled by the engine.
  Tensor t_logits;
  Tensor d_logits;

  bool keep_generating = true;

  // A buffer for tokens to be decoded (one at a time, per the Middleware's request)
  std::vector<int32_t> decode_buf(1, 0);

  // Decode new token.
  // Return true to continue generation, and false otherwise
  auto decode_token = [&](int32_t t) {
    decode_buf[0] = _last_tok = t;

    if (_ctx->is_eos(t)) {
      keep_generating = false;
      callback("", Sentence::END);
    } else {
      keep_generating = callback(_tokenizer->decode(decode_buf), Sentence::CONTINUE);
    }

    return keep_generating;
  };

  State::clear();

  auto& t_engine = *_engine["target"];
  auto& d_engine = *_engine["draft"];

  if (_n_past + tokens.size() > _ctx->size()) {
    __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
    callback("", Sentence::END);
    return true;
  }

  // Step 0: Process the prompt both on the target and draft models.
  bool d_pmpt, t_pmpt;
  if (_parallel) {
    std::thread dt([&]() { d_pmpt = d_engine.process(tokens, d_logits, false); });
    std::thread tt([&]() { t_pmpt = t_engine.process(tokens, t_logits, false); });
    dt.join();
    tt.join();
  } else {
    d_pmpt = d_engine.process(tokens, d_logits, false);
    t_pmpt = t_engine.process(tokens, t_logits, false);
  }

  if (!d_pmpt) return Dialog::abort("draft engine prompt processing failed", callback);
  if (!t_pmpt) return Dialog::abort("target engine prompt processing failed", callback);

  for (uint32_t idx = 0; idx < tokens.size(); idx++) {
    t_engine.updateTokenCheckpoint(tokens[idx], _n_past + idx);
    d_engine.updateTokenCheckpoint(tokens[idx], _n_past + idx);
  }
  // KV state Update
  _n_prompt += tokens.size();
  _n_past += tokens.size();

  if (!t_engine.updateKV(_n_past)) return Dialog::abort("target context size exceeded", callback);
  if (!d_engine.updateKV(_n_past)) return Dialog::abort("draft context size exceeded", callback);

  // Sample one token from the target.
  _last_tok = _t_sampler.process(t_logits);

  _kpis.prompt.update(start.elapsed_usec());

  // Log latest KPIs
  __KPIS("{}", kpis().dump(" "));

  if (!decode_token(_last_tok)) return true;

  // Done with the prompt, start generating
  start.reset();
  State::busy(true);

  processFollowOnGeneration(tokens, t_logits, d_logits, callback);

  State::busy(false);

  _kpis.generate.update(start.elapsed_usec());

  auto total_iteration = std::accumulate(_accepted_counts.begin(), _accepted_counts.end(), 0);
  auto accept_rate =
      float(_n_generated - 1) / total_iteration;  // -1: exclude first generated token
  _kpis.tps.tokenAcceptance = accept_rate;

  // Log latest KPIs in a single line
  __KPIS("{}", kpis().dump(" "));
  __KPIS("spec-dec: accepted counts: {}", _accepted_counts);

  return true;
}

void SpecDecDialog::reset() {
  Dialog::reset();
  _accepted_counts.clear();
}

// Registrator instance
static OnLoad regy([]() {
  Dialog::__register("spec-dec",
                     [](std::shared_ptr<Env> env, const std::string& name, const json& conf) {
                       return (Dialog*)new SpecDecDialog(env, name, conf);
                     });
});

void needSpdDialog() {}

}  // namespace qualla
