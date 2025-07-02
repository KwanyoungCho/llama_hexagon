//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef DIALOGCALLBACK_HPP
#define DIALOGCALLBACK_HPP

#include "qualla/detail/sentence.hpp"
#include "qualla/tokenizer.hpp"

namespace qualla {

typedef std::function<bool(const std::string&, Sentence::Code)> QueryCbFunction;
typedef std::function<bool(const int32_t*, const uint32_t, Sentence::Code)> TokenCbFunction;

typedef enum {
  QUALLA_CALLBACK_TYPE_TEXT      = 1,
  QUALLA_CALLBACK_TYPE_TOKEN     = 2,
  QUALLA_CALLBACK_TYPE_UNDEFINED = 0x7fffffff
} QuallaCallBackType;

class DialogCallback {
 public:
  DialogCallback() {
    m_basicQueryCb = std::make_shared<QueryCbFunction>();
    m_basicTokenCb = std::make_shared<TokenCbFunction>();
  }

  DialogCallback(QuallaCallBackType cbType) {
    if (cbType == QUALLA_CALLBACK_TYPE_TEXT) {
      m_basicQueryCb.reset(new QueryCbFunction);
      m_callBackType = QUALLA_CALLBACK_TYPE_TEXT;
    } else if (cbType == QUALLA_CALLBACK_TYPE_TOKEN) {
      m_basicTokenCb.reset(new TokenCbFunction);
      m_callBackType = QUALLA_CALLBACK_TYPE_TOKEN;
    }
  }

  bool callBack(const int32_t* tokens,
                const uint32_t sizeOfTokens,
                Sentence::Code scode,
                Tokenizer& _tokenizer) {
    if (m_callBackType == QUALLA_CALLBACK_TYPE_TEXT) {
      if (tokens) {
        std::vector<int32_t> tokenVec(tokens, tokens + sizeOfTokens);
        std::string outString = _tokenizer.decode(tokenVec);
        return (*m_basicQueryCb)(outString, scode);
      }
      return (*m_basicQueryCb)("", scode);
    } else if (m_callBackType == QUALLA_CALLBACK_TYPE_TOKEN) {
      return (*m_basicTokenCb)(tokens, sizeOfTokens, scode);
    } else {
      return false;
    }
  }

  void setCallBackType(QuallaCallBackType CbType) { m_callBackType = CbType; }

  QuallaCallBackType getCallBackType() { return m_callBackType; }

  std::shared_ptr<QueryCbFunction> getQueryCbFunc() { return m_basicQueryCb; }

  std::shared_ptr<TokenCbFunction> getTokenCbFunc() { return m_basicTokenCb; }

 private:
  QuallaCallBackType m_callBackType;
  std::shared_ptr<TokenCbFunction> m_basicTokenCb;
  std::shared_ptr<QueryCbFunction> m_basicQueryCb;
};
}  // namespace qualla
#endif  // DIALOGCALLBACK_HPP
