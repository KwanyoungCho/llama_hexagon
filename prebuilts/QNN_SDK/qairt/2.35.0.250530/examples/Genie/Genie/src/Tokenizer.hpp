//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once
#include <memory>

#include "GenieTokenizer.h"
#include "Util/HandleManager.hpp"
#include "qualla/env.hpp"
#include "qualla/tokenizer.hpp"

namespace genie {
class Tokenizer {
 public:
  static GenieTokenizer_Handle_t add(std::shared_ptr<Tokenizer> tokenizer);
  static std::shared_ptr<Tokenizer> get(GenieTokenizer_Handle_t handle);
  static void remove(GenieTokenizer_Handle_t handle);

  Tokenizer(std::reference_wrapper<qualla::Tokenizer>& quallaTokenizer);

  uint32_t encode(const char* inputString);
  uint32_t decode(const int32_t* tokenIds, const uint32_t numTokenIds);
  void getEncodedTokenIds(const int32_t** tokenIds, const int32_t allocatedSize);
  void getDecodedString(const char** outputString, const int32_t allocatedSize);

 private:
  static qnn::util::HandleManager<Tokenizer> s_manager;
  std::reference_wrapper<qualla::Tokenizer> m_quallaTokenizer;
  std::vector<int32_t> m_encodedTokenIds;
  std::string m_decodedString;
};

}  // namespace genie
