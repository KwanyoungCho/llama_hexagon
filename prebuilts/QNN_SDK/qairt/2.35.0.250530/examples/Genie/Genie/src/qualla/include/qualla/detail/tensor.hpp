//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_DETAIL_TENSOR_HPP
#define QUALLA_DETAIL_TENSOR_HPP

#include <cstddef>

namespace qualla {

enum TensorDataType {
  TENSOR_DATATYPE_UFIXED_POINT_8  = 0x01,
  TENSOR_DATATYPE_UFIXED_POINT_16 = 0x02,
  TENSOR_DATATYPE_FLOAT_POINT_16  = 0x03,
  TENSOR_DATATYPE_FLOAT_32        = 0x04,
  TENSOR_DATATYPE_UNKNOWN         = 0xFF
};

typedef struct {
  double scale;
  int32_t offset;
} TensorQuantizationParams;

/*
 *  This class has 2 uses: one where it solely points to logit data but doesn't own it,
 *  and one where it owns logit data in a vector of float and points to it using the same
 *  void* pointer.
 */
class Tensor {
 private:
  void* data                                  = nullptr;
  TensorDataType dataType                     = TENSOR_DATATYPE_UNKNOWN;
  TensorQuantizationParams quantizationParams = {1, 0};
  size_t numElements                          = 0;

 public:
  std::vector<float> logits;

  void* getData() { return this->data; }
  void setData(void* data) { this->data = data; }

  TensorDataType getDataType() { return this->dataType; }
  void setDataType(TensorDataType dataType) { this->dataType = dataType; }

  size_t getSize() { return this->numElements; }
  void setSize(size_t numElements) { this->numElements = numElements; }

  TensorQuantizationParams getQuantizationParams() { return this->quantizationParams; }
  void setQuantizationParams(double scale, int32_t offset) {
    this->quantizationParams.scale  = scale;
    this->quantizationParams.offset = offset;
  }

  Tensor getIndexedTensor(size_t index, size_t vocab, bool dynamicExtent = false) {
    size_t bytewidth = 0;

    switch (this->getDataType()) {
      case TENSOR_DATATYPE_UFIXED_POINT_8: {
        bytewidth = 1;
        break;
      }
      case TENSOR_DATATYPE_UFIXED_POINT_16: {
        bytewidth = 2;
        break;
      }
      case TENSOR_DATATYPE_FLOAT_POINT_16: {
        bytewidth = 2;
        break;
      }
      case TENSOR_DATATYPE_FLOAT_32: {
        bytewidth = 4;
        break;
      }
      default: {
        // throw error
        break;
      }
    }

    Tensor toReturn;

    auto returnData = (uint8_t*)(this->getData());
    returnData += index * vocab * bytewidth;
    toReturn.setData((void*)returnData);

    toReturn.setDataType(this->getDataType());

    if (!dynamicExtent)
      toReturn.setSize(vocab);
    else
      toReturn.setSize((this->numElements) - (index * vocab * bytewidth));

    toReturn.setQuantizationParams(this->quantizationParams.scale, this->quantizationParams.offset);
    return toReturn;
  }
};

}  // namespace qualla
#endif  // QUALLA_DETAIL_TENSOR_HPP