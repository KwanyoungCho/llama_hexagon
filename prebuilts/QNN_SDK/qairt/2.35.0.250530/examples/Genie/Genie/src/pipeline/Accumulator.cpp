//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <iostream>
#include <memory>

#include "Accumulator.hpp"
using namespace genie;

Accumulator::Accumulator(size_t bufferSize) { embeddingsBuffer.reserve(bufferSize); }

bool Accumulator::append(uint8_t* data, size_t dataSize) {
  size_t startAddress = embeddingsBuffer.size();
  embeddingsBuffer.insert(embeddingsBuffer.end(), data, data + dataSize);

  return true;
}

bool Accumulator::append(
    void* src, std::string srcDataType, double srcScale, int32_t srcOffset, size_t numElements) {
  requantEmbedding(src, srcDataType, srcScale, srcOffset, numElements);
  return true;
}

bool Accumulator::flush() {
  embeddingsBuffer.clear();
  return true;
}

void* Accumulator::getData() { return embeddingsBuffer.data(); }

size_t Accumulator::getDataSize() { return embeddingsBuffer.size(); }

void Accumulator::setEncoding(std::string dType,
                              double generatorScale,
                              int32_t generatorOffset,
                              size_t generatorByteWidth) {
  byteWidth = generatorByteWidth;
  offset    = generatorOffset;
  scale     = generatorScale;
  dataType  = dType;
}

void Accumulator::requantEmbedding(
    void* src, std::string srcDataType, double srcScale, int32_t srcOffset, size_t length) {
  double requantScale    = srcScale / scale;
  int32_t requantOffset  = srcOffset * requantScale - offset;
  size_t dstStartAddress = embeddingsBuffer.size() / byteWidth;
  size_t embeddingSize   = length * byteWidth;
  embeddingsBuffer.resize(embeddingsBuffer.size() + embeddingSize);
  void* dst = embeddingsBuffer.data();
  for (int i = 0; i < length; i++) {
    if (srcDataType == "QNN_DATATYPE_SFIXED_POINT_8" && dataType == "QNN_DATATYPE_SFIXED_POINT_8") {
      static_cast<int8_t*>(dst)[i + dstStartAddress] =
          static_cast<int8_t>(requantScale * static_cast<int8_t*>(src)[i] + requantOffset);
    } else if (srcDataType == "QNN_DATATYPE_SFIXED_POINT_8" &&
               dataType == "QNN_DATATYPE_SFIXED_POINT_16") {
      static_cast<int16_t*>(dst)[i + dstStartAddress] =
          static_cast<int16_t>(requantScale * static_cast<int8_t*>(src)[i] + requantOffset);
    } else if (srcDataType == "QNN_DATATYPE_UFIXED_POINT_8" &&
               dataType == "QNN_DATATYPE_UFIXED_POINT_8") {
      static_cast<uint8_t*>(dst)[i + dstStartAddress] =
          static_cast<uint8_t>(requantScale * static_cast<uint8_t*>(src)[i] + requantOffset);
    } else if (srcDataType == "QNN_DATATYPE_UFIXED_POINT_8" &&
               dataType == "QNN_DATATYPE_UFIXED_POINT_16") {
      static_cast<uint16_t*>(dst)[i + dstStartAddress] =
          static_cast<uint16_t>(requantScale * static_cast<uint8_t*>(src)[i] + requantOffset);
    } else if (srcDataType == "QNN_DATATYPE_SFIXED_POINT_16" &&
               dataType == "QNN_DATATYPE_SFIXED_POINT_8") {
      static_cast<int8_t*>(dst)[i + dstStartAddress] =
          static_cast<int8_t>(requantScale * static_cast<int16_t*>(src)[i] + requantOffset);
    } else if (srcDataType == "QNN_DATATYPE_SFIXED_POINT_16" &&
               dataType == "QNN_DATATYPE_SFIXED_POINT_16") {
      static_cast<int16_t*>(dst)[i + dstStartAddress] =
          static_cast<int16_t>(requantScale * static_cast<int16_t*>(src)[i] + requantOffset);
    } else if (srcDataType == "QNN_DATATYPE_UFIXED_POINT_16" &&
               dataType == "QNN_DATATYPE_UFIXED_POINT_8") {
      static_cast<uint8_t*>(dst)[i + dstStartAddress] =
          static_cast<uint8_t>(requantScale * static_cast<uint16_t*>(src)[i] + requantOffset);
    } else if (srcDataType == "QNN_DATATYPE_UFIXED_POINT_16" &&
               dataType == "QNN_DATATYPE_UFIXED_POINT_16") {
      static_cast<uint16_t*>(dst)[i + dstStartAddress] =
          static_cast<uint16_t>(requantScale * static_cast<uint16_t*>(src)[i] + requantOffset);
      // Quantize float to fixed point
    } else if (srcDataType == "QNN_DATATYPE_FLOAT_32" &&
               dataType == "QNN_DATATYPE_UFIXED_POINT_16") {
      static_cast<uint16_t*>(dst)[i + dstStartAddress] =
          static_cast<uint16_t>(((float*)src)[i] / scale - offset);
    } else if (srcDataType == "QNN_DATATYPE_FLOAT_32" &&
               dataType == "QNN_DATATYPE_UFIXED_POINT_8") {
      static_cast<uint8_t*>(dst)[i + dstStartAddress] =
          static_cast<uint8_t>(((float*)src)[i] / scale - offset);
    } else if (srcDataType == "QNN_DATATYPE_FLOAT_32" &&
               dataType == "QNN_DATATYPE_SFIXED_POINT_16") {
      static_cast<int16_t*>(dst)[i + dstStartAddress] =
          static_cast<int16_t>(((float*)src)[i] / scale - offset);
    } else if (srcDataType == "QNN_DATATYPE_FLOAT_32" &&
               dataType == "QNN_DATATYPE_SFIXED_POINT_8") {
      static_cast<int8_t*>(dst)[i + dstStartAddress] =
          static_cast<int8_t>(((float*)src)[i] / scale - offset);
      // Dequantize fixed point to float
    } else if (srcDataType == "QNN_DATATYPE_UFIXED_POINT_16" &&
               dataType == "QNN_DATATYPE_FLOAT_32") {
      static_cast<float*>(dst)[i + dstStartAddress] =
          static_cast<float>(srcScale * (static_cast<uint16_t*>(src)[i] + srcOffset));
    } else if (srcDataType == "QNN_DATATYPE_UFIXED_POINT_8" &&
               dataType == "QNN_DATATYPE_FLOAT_32") {
      static_cast<float*>(dst)[i + dstStartAddress] =
          static_cast<float>(srcScale * (static_cast<uint8_t*>(src)[i] + srcOffset));
    } else if (srcDataType == "QNN_DATATYPE_SFIXED_POINT_16" &&
               dataType == "QNN_DATATYPE_FLOAT_32") {
      static_cast<float*>(dst)[i + dstStartAddress] =
          static_cast<float>(srcScale * (static_cast<int16_t*>(src)[i] + srcOffset));
    } else if (srcDataType == "QNN_DATATYPE_SFIXED_POINT_8" &&
               dataType == "QNN_DATATYPE_FLOAT_32") {
      static_cast<float*>(dst)[i + dstStartAddress] =
          static_cast<float>(srcScale * (static_cast<int8_t*>(src)[i] + srcOffset));
    } else if (srcDataType == "QNN_DATATYPE_FLOAT_32" && dataType == "QNN_DATATYPE_FLOAT_32") {
      static_cast<float*>(dst)[i + dstStartAddress] = static_cast<float*>(src)[i];
    } else {
      throw Exception(GENIE_STATUS_ERROR_GENERAL, "unsupported requant operation");
    }
  }
}