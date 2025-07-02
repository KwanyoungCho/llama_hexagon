//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_DETAIL_SENTENCE_HPP
#define QUALLA_DETAIL_SENTENCE_HPP

#include <string>

namespace qualla {

struct Sentence {
  enum Code {
    COMPLETE,  // Complete sentence
    BEGIN,     // First part of the sentence
    CONTINUE,  // Continuation of the sentense
    END,       // Last part of the sentence
    ABORT,     // Sentence aborted
    REWIND     // KV rewind as per prefix match before processing Query
  };

  static inline std::string str(Code c) {
    static const char* s[]{"COMPLETE", "BEGIN", "CONTINUE", "END", "ABORT"};
    return std::string(s[c]);
  }
};

}  // namespace qualla

#endif  // QUALLA_DETAIL_SENTENCE_HPP
