//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
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

#include "qualla/detail/json.hpp"
#include "qualla/dialog.hpp"
#include "qualla/env.hpp"

namespace qualla {

class BasicDialog : public Dialog {
 public:
  BasicDialog(std::shared_ptr<Env> env, const std::string& name, const json& conf);

  virtual bool process(std::vector<int32_t>& tokens, qualla::DialogCallback callback) override;

  virtual bool process(std::vector<int32_t>& tokens, Dialog::Callback callback) override;

  virtual bool process(std::vector<uint8_t>& embedding_vectors,
                       Dialog::T2ECallback t2eCallback,
                       Dialog::Callback callback) override;

 protected:
  virtual bool supportsLongContext() const override { return true; };

 private:
  bool processFollowOnGeneration(std::vector<int32_t>& tokens,
                                 Tensor& logits,
                                 Dialog::Callback callback);

  bool processFollowOnGeneration(std::vector<int32_t>& tokens,
                                 Tensor& logits,
                                 qualla::DialogCallback callback);
};

}  // namespace qualla

#endif  // QUALLA_DETAIL_BASIC_DIALOG_HPP
