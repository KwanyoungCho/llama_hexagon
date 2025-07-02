//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_DETAIL_BASIC_DIALOG_HPP
#define QUALLA_DETAIL_BASIC_DIALOG_HPP

#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include "qualla/detail/config.hpp"
#include "qualla/detail/json.hpp"
#include "qualla/dialog.hpp"
#include "qualla/env.hpp"

namespace qualla {

using qc = qualla::Config;

class MultiStreamDialog : public Dialog {
 public:
  MultiStreamDialog(std::shared_ptr<Env> env, const std::string& name, const json& conf)
      : Dialog(env, name, conf) {
    _vocab       = _ctx->n_vocab();
    _n_streams   = qc::optional<int32_t>(conf, "n-streams", 1);
    _p_threshold = qc::optional<float>(conf, "p-threshold", 0.0);
  }

  virtual bool process(std::vector<int32_t>& tokens, Dialog::Callback callback) override;

  virtual bool process(std::vector<uint8_t>& embedding_vectors,
                       Dialog::T2ECallback t2eCallback,
                       Dialog::Callback callback) override;

  virtual bool process(std::vector<int32_t>& tokens, DialogCallback callback) override {
    return false;
  }

 protected:
  int32_t _vocab;
  int32_t _n_streams;
  int32_t _prompt_len{-1};
  float _p_threshold;

 private:
  bool processFollowOnGeneration(std::vector<std::vector<int32_t>>& streams,
                                 Tensor& logits,
                                 Dialog::Callback callback);
};

}  // namespace qualla

#endif  // QUALLA_DETAIL_BASIC_DIALOG_HPP
