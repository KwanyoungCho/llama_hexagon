//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#include <exception>
#include <set>

#include "Exception.hpp"
#include "Macro.hpp"
#include "Tokenizer.hpp"
#include "qualla/detail/json.hpp"
#if ENABLE_DEBUG_LOGS
#include <iostream>
#endif

using namespace genie;

//=============================================================================
// Tokenizer functions
//=============================================================================

qnn::util::HandleManager<Tokenizer> Tokenizer::s_manager;

GenieTokenizer_Handle_t Tokenizer::add(std::shared_ptr<Tokenizer> tokenizer) {
  return (GenieTokenizer_Handle_t)s_manager.add(tokenizer);
}

std::shared_ptr<Tokenizer> Tokenizer::get(GenieTokenizer_Handle_t handle) {
  return s_manager.get((qnn::util::Handle_t)handle);
}

void Tokenizer::remove(GenieTokenizer_Handle_t handle) {
  s_manager.remove((qnn::util::Handle_t)handle);
}

Tokenizer::Tokenizer(std::reference_wrapper<qualla::Tokenizer>& quallaTokenizer)
    : m_quallaTokenizer(quallaTokenizer) {}

uint32_t Tokenizer::encode(const char* inputString) {
  if (!inputString) return 0;
  std::string inputStr = inputString;
  return m_quallaTokenizer.get().encode(inputStr, m_encodedTokenIds);
}

uint32_t Tokenizer::decode(const int32_t* tokenIds, const uint32_t numTokenIds) {
  if (!tokenIds || !numTokenIds) return 0;
  std::vector<int32_t> tokenVec(tokenIds, tokenIds + numTokenIds);
  m_decodedString = m_quallaTokenizer.get().decode(tokenVec);
  return m_decodedString.size() + 1;
}

void Tokenizer::getEncodedTokenIds(const int32_t** tokenIds, const int32_t allocatedSize) {
  if (m_encodedTokenIds.size() * sizeof(int32_t) != allocatedSize) {
    throw Exception(GENIE_STATUS_ERROR_MEM_ALLOC, "Encoded TokenIds buffer size mismatch.");
  }
  memcpy((void*)*tokenIds,
         (void*)m_encodedTokenIds.data(),
         m_encodedTokenIds.size() * sizeof(int32_t));
  m_encodedTokenIds.clear();
}

void Tokenizer::getDecodedString(const char** outputString, const int32_t allocatedSize) {
  if (m_decodedString.size() + 1 != allocatedSize) {
    throw Exception(GENIE_STATUS_ERROR_MEM_ALLOC, "Decoded String buffer size mismatch.");
  }
  memcpy((void*)*outputString, (void*)m_decodedString.c_str(), m_decodedString.size());
  ((char*)(*outputString))[m_decodedString.length()] = '\0';
  m_decodedString.clear();
}
