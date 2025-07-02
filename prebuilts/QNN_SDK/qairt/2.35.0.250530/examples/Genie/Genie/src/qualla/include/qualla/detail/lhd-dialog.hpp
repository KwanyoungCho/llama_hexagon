//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_DETAIL_LOOKAHEAD_DIALOG_HPP
#define QUALLA_DETAIL_LOOKAHEAD_DIALOG_HPP

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "qualla/detail/json.hpp"
#include "qualla/dialog.hpp"
#include "qualla/env.hpp"

namespace qualla {

class LhdDecDialog : public Dialog {
 public:
  LhdDecDialog(std::shared_ptr<Env> env, const std::string& name, const json& conf);

  virtual bool process(std::vector<int32_t>& tokens, Dialog::Callback callback) override;

  virtual bool process(std::vector<int32_t>& tokens, DialogCallback callback) override {
    return false;
  }

 protected:
  virtual bool supportsLongContext() const override { return true; };

  enum LHFwdMode { ALWAYS_FWD_ONE = 0x0, FWD_MAX_HIT = 0x1, FWD_LEVEL = 0x2 };
  struct ngram_data {
    bool active    = false;
    int32_t seq_id = -1;

    // match pos
    std::vector<int> i_batch;
    std::vector<int32_t> tokens;
  };

  // n-gram pool
  struct ngram_container {
    ngram_container(int n_vocab, int n, int g) {
      cnt.resize(n_vocab);
      head.resize(n_vocab);
      tokens.resize(n_vocab * g * (n - 1));
    }

    int n_total = 0;

    std::vector<size_t> cnt;
    std::vector<int> head;

    // [n_vocab][G][N - 1]
    std::vector<int32_t> tokens;
  };

  // W/N/G
  size_t _window;
  size_t _ngram;
  size_t _gcap;

  size_t _n_accept{0};   // number of match tokens
  size_t _level_idx{1};  // lookahead branch level

  // lookahead branch update mode
  std::string _lhd_mode_str;
  LHFwdMode _lhd_update_mode{ALWAYS_FWD_ONE};

  // verification branch
  std::vector<ngram_data> v_branch;
  std::vector<std::vector<int32_t>> lhd_branch;
  std::vector<int32_t> lhd_branch_prev;

  std::vector<int32_t> batch;
  std::vector<int32_t> attention_map;
};

}  // namespace qualla

#endif  // QUALLA_DETAIL_LOOKAHEAD_DIALOG_HPP
