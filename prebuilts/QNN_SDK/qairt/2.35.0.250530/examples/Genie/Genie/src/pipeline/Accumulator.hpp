//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once
#ifndef ACCUMULATOR_HPP
#define ACCUMULATOR_HPP
#include <memory>
#include <string>
#include <vector>

#include "Exception.hpp"
#include "GenieCommon.h"

namespace genie {
class Accumulator {
 public:
  Accumulator(size_t bufferSize = 0);
  bool flush();
  bool append(uint8_t* data, size_t dataSizeize);
  bool append(
      void* src, std::string srcDataType, double srcScale, int32_t srcOffset, size_t numElements);
  void* getData();
  size_t getDataSize();
  std::string& getDataType() { return dataType; }
  double& getScale() { return scale; }
  int32_t& getOffset() { return offset; }
  size_t getByteWidth() { return byteWidth; }
  void setEncoding(std::string dType,
                   double generatorScale,
                   int32_t generatorOffset,
                   size_t generatorByteWidth);

 private:
  void requantEmbedding(
      void* src, std::string srcDataType, double srcScale, int32_t srcOffset, size_t length);

  std::string dataType{"QNN_DATATYPE_FLOAT_32"};
  double scale{1.0};
  int32_t offset{0};
  size_t byteWidth{4};
  std::vector<uint8_t> embeddingsBuffer;
  size_t accumulatorSize{0};
};
}  // namespace genie
#endif  // ACCUMULATOR_HPP