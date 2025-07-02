//=============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//=============================================================================

#include "Dialog.hpp"
#include "Exception.hpp"
#include "GenieTokenizer.h"
#include "Macro.hpp"
#include "Util/HandleManager.hpp"

using namespace genie;
GENIE_API
Genie_Status_t GenieTokenizer_encode(const GenieTokenizer_Handle_t tokenizerHandle,
                                     const char* inputString,
                                     const Genie_AllocCallback_t callback,
                                     const int32_t** tokenIds,
                                     uint32_t* numTokenIds) {
  try {
    GENIE_ENSURE(tokenizerHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(inputString, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    auto tokenizer = Tokenizer::get(tokenizerHandle);
    GENIE_ENSURE(tokenizer, GENIE_STATUS_ERROR_INVALID_HANDLE);
    *numTokenIds = tokenizer->encode(inputString);
    GENIE_ENSURE(*numTokenIds, GENIE_STATUS_ERROR_GENERAL);
    callback(*numTokenIds * sizeof(int32_t), (const char**)tokenIds);
    tokenizer->getEncodedTokenIds(tokenIds, *numTokenIds * sizeof(int32_t));
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}

GENIE_API
Genie_Status_t GenieTokenizer_decode(const GenieTokenizer_Handle_t tokenizerHandle,
                                     const int32_t* tokenIds,
                                     const uint32_t numTokenIds,
                                     const Genie_AllocCallback_t callback,
                                     const char** outputString) {
  try {
    GENIE_ENSURE(tokenizerHandle, GENIE_STATUS_ERROR_INVALID_HANDLE);
    GENIE_ENSURE(tokenIds, GENIE_STATUS_ERROR_INVALID_ARGUMENT);
    auto tokenizer = Tokenizer::get(tokenizerHandle);
    GENIE_ENSURE(tokenizer, GENIE_STATUS_ERROR_INVALID_HANDLE);
    const uint32_t outputSize = tokenizer->decode(tokenIds, numTokenIds);
    GENIE_ENSURE(outputSize, GENIE_STATUS_ERROR_GENERAL);
    callback(outputSize * sizeof(char), outputString);
    tokenizer->getDecodedString(outputString, outputSize * sizeof(char));
  } catch (const std::exception& e) {
    std::cerr << e.what() << std::endl;
    return GENIE_STATUS_ERROR_GENERAL;
  }
  return GENIE_STATUS_SUCCESS;
}
