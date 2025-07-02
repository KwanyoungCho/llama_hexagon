//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <cmath>
#include <cstdio>
#include <filesystem>
#include <functional>
#include <random>
#include <string>

#include "qualla/detail/config.hpp"
#include "qualla/detail/lhd-dialog.hpp"
#include "qualla/detail/onload.hpp"
#include "qualla/detail/timer.hpp"
#include "qualla/dialog.hpp"

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

LhdDecDialog::LhdDecDialog(std::shared_ptr<Env> env, const std::string& name, const json& conf)
    : Dialog(env, name, conf) {
  _window = qc::optional<size_t>(conf, "window", 8);
  _ngram  = qc::optional<size_t>(conf, "ngram", 3);
  _gcap   = qc::optional<size_t>(conf, "gcap", 8);

  _lhd_mode_str = qc::optional<std::string>(conf, "lhd-update-mode", "ALWAYS_FWD_ONE");
}

bool LhdDecDialog::process(std::vector<int32_t>& tokens, Dialog::Callback callback) {
  // Check for prev failures and bail out early
  if (State::failed()) return false;

  Timer start;

  // Vector for storing logits.
  // Allocated & filled by the engine.
  Tensor logits;
  std::vector<int32_t> resultTokens;

  State::clear();

  auto& sampler = *_sampler["primary"];
  auto& engine  = *_engine["primary"];

  using FF = Engine::Feature::Flags;
  if (engine.supports(FF::DYNAMIC_LOAD)) engine.load();

  if (_n_past + tokens.size() > _ctx->size()) {
    __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
    callback("", Sentence::END);
    return true;
  }

  if (!engine.process(tokens, logits, false))
    return Dialog::abort("engine prompt processing failed", callback);

  _n_prompt += tokens.size();
  _n_past += tokens.size();

  if (!engine.updateKV(_n_past)) return Dialog::abort("context size exceeded", callback);

  std::vector<int32_t> tokens_tmp(1);
  tokens_tmp[0] = _last_tok = sampler.process(logits);
  resultTokens.push_back(_last_tok);

  _n_generated++;

  _kpis.prompt.update(start.elapsed_usec());

  // Log latest KPIs
  __KPIS("{}", kpis().dump(" "));

  if (_ctx->is_eos(_last_tok)) {
    callback("", Sentence::END);
    return true;
  }

  // Exit condition : Prediction limit reached OR ctx size limit reached
  if (!callback(_tokenizer->decode(tokens_tmp), Sentence::BEGIN)) return true;

  State::busy(true);

  // verification branch init
  v_branch.resize(_gcap);

  // n-gram pools
  const size_t n_vocab = _ctx->n_vocab();
  ngram_container ngrams_pool(n_vocab, _ngram, _gcap);

  // lookahead branch first level init
  lhd_branch.resize(_ngram - 1);
  lhd_branch_prev.resize(_window);

  for (int j = 0; j < _ngram - 1; j++) {
    lhd_branch[j].resize(_window);

    for (int i = 0; i < _window; i++) {
      if (j == 0) {
        // initialize with the random token from prompt
        lhd_branch[j][i] = tokens[1 + rand() % (tokens.size() - 1)];
      } else {
        // initialize with a sequence of increasing numbers
        lhd_branch[j][i] = 1000 + i;
      }
    }
  }

  // lookahead branch other level init
  while (_level_idx < _ngram - 1) {
    batch.clear();
    attention_map.clear();

    // fill the first token of the first level
    batch.push_back(_last_tok);
    attention_map.push_back(-1);
    lhd_branch[0][0] = _last_tok;

    // fill the remaining WINDOW - 1 tokens for the first level
    for (int i = 1; i < _window; i++) {
      batch.push_back(lhd_branch[0][i]);
      attention_map.push_back(i - 1);
    }

    // fill the rest of the levels
    for (int j = 1; j < _ngram - 1; j++) {
      for (int i = 0; i < _window; i++) {
        batch.push_back(lhd_branch[j][i]);
        attention_map.push_back((j - 1) * _window + i);
      }
    }

    // re-init tokens batch
    tokens.resize(_window * (_ngram - 1));
    tokens = batch;

    if (_n_past + tokens.size() > _ctx->size()) {
      __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
      callback("", Sentence::END);
      break;
    }

    size_t n_tok = engine.process(tokens, attention_map, logits, true);
    if (n_tok != tokens.size())
      return Dialog::abort("engine lookahead branch processing failed", callback);

    for (int i = 0; i < _window; i++) {
      size_t sample_tmp_idx = (_level_idx - 1) * _window + i;
      // sampler from logits all
      Tensor indexedLogits      = logits.getIndexedTensor(sample_tmp_idx, n_vocab);
      int32_t sampled_tmp_token = sampler.process(indexedLogits);
      lhd_branch[_level_idx][i] = sampled_tmp_token;
    }

    _level_idx++;
  }

  if (_lhd_mode_str == "FWD_MAX_HIT")
    _lhd_update_mode = FWD_MAX_HIT;
  else if (_lhd_mode_str == "FWD_LEVEL")
    _lhd_update_mode = FWD_LEVEL;
  else
    _lhd_update_mode = ALWAYS_FWD_ONE;

  size_t iterationCount = 0;

  start.reset();

  while (true) {
    if (State::canceled()) {
      callback("", Sentence::END);
      break;
    }
    // input batch init
    {
      batch.clear();
      attention_map.clear();

      // fill the first token of the first level
      batch.push_back(_last_tok);
      attention_map.push_back(-1);
      // lhd_branch[0][0] = _last_tok;

      // fill the remaining WINDOW - 1 tokens for the first level
      for (int i = 1; i < _window; i++) {
        batch.push_back(lhd_branch[0][i]);
        attention_map.push_back(i - 1);
      }

      // fill the rest of the levels
      for (int j = 1; j < _ngram - 1; j++) {
        for (int i = 0; i < _window; i++) {
          batch.push_back(lhd_branch[j][i]);
          attention_map.push_back((j - 1) * _window + i);
        }
      }

      // build verification n-grams(branch)
      {
        const int g_cur = ngrams_pool.cnt[_last_tok];

        v_branch.resize(g_cur);
        // input_token_batch.size = (_window + g_cur) * (_ngram - 1);
        tokens.resize((_window + g_cur) * (_ngram - 1));
        for (int g = 0; g < g_cur; g++) {
          v_branch[g].active = true;
          v_branch[g].tokens.resize(_ngram);
          v_branch[g].i_batch.resize(_ngram);
          v_branch[g].seq_id     = _window + 1 + g;
          v_branch[g].i_batch[0] = 0;
          v_branch[g].tokens[0]  = _last_tok;
        }

        for (int j = 0; j < _ngram - 1; j++) {
          for (int g = 0; g < g_cur; g++) {
            const int idx              = _last_tok * (_ngram - 1) * _gcap + g * (_ngram - 1);
            const int32_t t            = ngrams_pool.tokens[idx + j];
            v_branch[g].tokens[j + 1]  = t;
            v_branch[g].i_batch[j + 1] = j + 1;
          }
        }

        for (int g = 0; g < g_cur; g++) {
          for (int j = 0; j < _ngram - 1; j++) {
            batch.push_back(v_branch[g].tokens[j + 1]);
            if (j == 0)
              attention_map.push_back(0);
            else
              attention_map.push_back(batch.size() - 2);
          }
        }
      }
    }

    // re-init tokens batch
    std::vector<bool> selected(attention_map.size(), false);
    tokens = batch;

    if (_n_past + tokens.size() > _ctx->size()) {
      __WARN("Context limit exceeded ({} + {} > {})", _n_past, tokens.size(), _ctx->size());
      callback("", Sentence::END);
      break;
    }

    size_t n_tok = engine.process(tokens, attention_map, logits, true);
    if (n_tok != tokens.size()) return Dialog::abort("engine gen processing failed", callback);
    iterationCount++;

    // verification branch seq-id
    size_t seq_id_best = 0;
    // max hit pos
    size_t i_batch_best = 0;

    // Lookahead decoding and verification
    for (int v = 0; v < _ngram; ++v) {
      int i_batch = 0;

      if (v > 0) {
        for (int g = 0; g < (int)v_branch.size(); g++) {
          // record the best matched seq and pos
          if (v_branch[g].active) {
            i_batch      = v_branch[g].i_batch[v];
            i_batch_best = i_batch;
            seq_id_best  = v_branch[g].seq_id;
            ++_n_accept;
            break;
          }
        }

        if (i_batch == 0) {
          break;
        }
      }

      size_t sample_idx;
      if (seq_id_best == 0)
        sample_idx = 0;
      else
        sample_idx =
            _window * (_ngram - 1) + (seq_id_best - (_window + 1)) * (_ngram - 1) + i_batch - 1;

      // vector selected set
      selected[sample_idx] = true;

      // sampler from logits all
      Tensor sample_logit = logits.getIndexedTensor(sample_idx, n_vocab);
      _last_tok           = sampler.process(sample_logit);

      std::vector<int32_t> tokens_tmp(1);
      tokens_tmp[0] = _last_tok;

      resultTokens.push_back(_last_tok);
      _n_generated++;
      _n_past++;

      if (_ctx->is_eos(_last_tok)) break;

      if (!callback(_tokenizer->decode(tokens_tmp), Sentence::CONTINUE)) {
        if (!engine.updateKV(_n_past, selected))  // update kv to latest before returning
          return Dialog::abort("context size exceeded", callback);
        return true;
      }

      // if verify pass, check the next sample token until verifing failed
      for (int g = 0; g < (int)v_branch.size(); g++) {
        // update the n-gram active status
        if (v_branch[g].active) {
          if (v == _ngram - 1) {
            v_branch[g].active = false;
          } else {
            if (_last_tok != v_branch[g].tokens[v + 1]) {
              v_branch[g].active = false;
            }
          }
        }
      }

      // update lookahead tokens when v=0 OR verify match
      {
        for (int i = 0; i < _window; i++) {
          lhd_branch_prev[i] = lhd_branch[0][i];
        }

        if (v == 0) {
          for (int j = 0; j < _ngram - 2; j++) {
            lhd_branch[j] = lhd_branch[j + 1];
          }

          // sample from the last level
          for (int i = 0; i < _window; i++) {
            size_t sample_idx         = (_ngram - 2) * _window + i;
            Tensor sample_logit       = logits.getIndexedTensor(sample_idx, n_vocab);
            lhd_branch[_ngram - 2][i] = sampler.process(sample_logit);
          }
        } else {
          if (_lhd_update_mode == FWD_MAX_HIT) {
            // update lookahead branch by foward
            for (int j = 0; j < _ngram - 1; j++) {
              for (int i = 0; i < _window - v; i++) {
                lhd_branch[j][i] = lhd_branch[j][i + 1];
              }
            }
          } else if (_lhd_update_mode == FWD_LEVEL) {
            // update lookahead branch by shifting level
            for (int j = 0; j < _ngram - 2; j++) {
              lhd_branch[j] = lhd_branch[j + 1];
            }

            for (int i = 0; i < _window; i++) {
              // init from the previous level
              lhd_branch[_ngram - 2][i] = lhd_branch[0][i];
            }
          }
        }
      }

      // update n-grams pool
      // only update n-grams pools when v=0
      if (v == 0) {
        std::vector<int32_t> ngram(_ngram - 1);
        // n-gram pool generation
        for (int f = 0; f < _window; ++f) {
          const int ft = lhd_branch_prev[f];  // first token of the n-gram

          for (int j = 0; j < _ngram - 1; ++j) {
            ngram[j] = lhd_branch[j][f];
          }

          // filter-out repeating n-grams
          {
            bool is_unique = true;

            for (int k = 0; k < ngrams_pool.cnt[ft]; ++k) {
              // caculate the related idx by the first n-gram token
              const int idx = ft * (_ngram - 1) * _gcap + k * (_ngram - 1);

              bool is_match = true;
              for (int j = 0; j < _ngram - 1; ++j) {
                if (ngrams_pool.tokens[idx + j] != ngram[j]) {
                  is_match = false;
                  break;
                }
              }

              // if n-gram match all, discard one of them
              if (is_match) {
                is_unique = false;
                break;
              }
            }

            if (!is_unique) {
              continue;
            }
          }

          const int head = ngrams_pool.head[ft];
          const int idx  = ft * (_ngram - 1) * _gcap + head * (_ngram - 1);

          for (int i = 0; i < _ngram - 1; i++) {
            // update the n-gram pool with new n-gram
            ngrams_pool.tokens[idx + i] = ngram[i];
          }

          ngrams_pool.cnt[ft]  = std::min(_gcap, ngrams_pool.cnt[ft] + 1);
          ngrams_pool.head[ft] = (head + 1) % _gcap;

          ngrams_pool.n_total++;
        }
      }
    }

    if (_lhd_update_mode == FWD_MAX_HIT) {
      // std::random_device rd;
      // std::mt19937 gen(rd());
      // std::uniform_int_distribution<> dis(0, resultTokens.size() - 1);

      // fill lookahead branch
      for (int i = 0; i < _ngram - 1; i++) {
        for (int j = _window - i_batch_best; j < _window; j++) {
          lhd_branch[i][j] = resultTokens[1 + rand() % (resultTokens.size() - 1)];
          // lhd_branch[i][j] = resultTokens[dis(gen)];
          // std::cout << "Fill token = " << lhd_branch[i][j] << std::endl;
        }
      }
    }

    // KV cache management
    if (!engine.updateKV(_n_past, selected))
      return Dialog::abort("context size exceeded", callback);

    if (_ctx->is_eos(_last_tok)) {
      callback("", Sentence::END);
      break;
    }
  }

  State::busy(false);

  _kpis.generate.update(start.elapsed_usec());
  _kpis.tps.tokenAcceptance =
      float(_n_generated - 1) / iterationCount;  // -1: exclude first generated token

  // Log latest KPIs in a single line
  __KPIS("{}", kpis().dump(" "));
  std::cout << std::endl << std::endl << std::flush;
  __DEBUG("lhd-dec: n_generated = {} ---------- n_accept = {}", _n_generated, _n_accept);

  return !State::failed();
}

// Registrator instance
static OnLoad regy([]() {
  Dialog::__register("lhd-dec",
                     [](std::shared_ptr<Env> env, const std::string& name, const json& conf) {
                       return (Dialog*)new LhdDecDialog(env, name, conf);
                     });
});

void needLadeDialog() {}

}  // namespace qualla
