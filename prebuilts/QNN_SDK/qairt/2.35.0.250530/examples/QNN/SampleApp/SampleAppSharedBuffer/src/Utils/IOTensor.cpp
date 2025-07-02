//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All rights reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#include <algorithm>
#include <cstring>
#include <fstream>
#include <iostream>

#include "DataUtil.hpp"
#include "IOTensor.hpp"
#include "Logger.hpp"
#ifndef __hexagon__
#include "PAL/Directory.hpp"
#include "PAL/FileOp.hpp"
#include "PAL/Path.hpp"
#endif
#include "PAL/DynamicLoading.hpp"
#include "PAL/StringOp.hpp"
#include "QnnMem.h"
#include "QnnTypeMacros.hpp"
using namespace qnn;
using namespace qnn::tools;

iotensor::IOTensor::IOTensor(QNN_INTERFACE_VER_TYPE* qnnInterface, bool sharedBuffers)
    : m_batchSize(1), m_numFilesPopulated(0) {
  if (nullptr != qnnInterface) {
    m_libCdspRpc = pal::dynamicloading::dlOpen(
#ifdef _WIN32
        pal::Path::combine(pal::Dsp::getDspDriverPath(), "libcdsprpc.dll").c_str(),
#else
        "libcdsprpc.so",
#endif
        pal::dynamicloading::DL_NOW | pal::dynamicloading::DL_LOCAL);
    if (nullptr == m_libCdspRpc) {
      QNN_ERROR("Unable to load backend. dlerror(): %s", pal::dynamicloading::dlError());
    }

    m_rpcMemAlloc = (RpcMemAllocFn_t)pal::dynamicloading::dlSym(m_libCdspRpc, "rpcmem_alloc");
    m_rpcMemFree  = (RpcMemFreeFn_t)pal::dynamicloading::dlSym(m_libCdspRpc, "rpcmem_free");
    m_rpcMemToFd  = (RpcMemToFdFn_t)pal::dynamicloading::dlSym(m_libCdspRpc, "rpcmem_to_fd");

    if (nullptr == m_rpcMemAlloc || nullptr == m_rpcMemFree || nullptr == m_rpcMemToFd) {
      QNN_ERROR("Unable to access symbols in sharedbuffer.dlerror():%s",
                pal::dynamicloading::dlError());
      return;
    }

    m_useSharedBuffer = sharedBuffers;
    m_qnnInterface    = qnnInterface;
    QNN_INFO("Init sharebuffer IOTensor success.");

  } else {
    m_useSharedBuffer = sharedBuffers;
    m_qnnInterface    = nullptr;
    QNN_WARN("Init sharebuffer IOTensor fail.");
  }
}

iotensor::StatusCode iotensor::IOTensor::getContextInfo(Qnn_ContextHandle_t* context) {
  m_context = context;
  return StatusCode::SUCCESS;
}

void* iotensor::IOTensor::getTensorBuffer(Qnn_Tensor_t* tensor) {
  if (nullptr == tensor) {
    QNN_WARN("Received a nullpointer to a tensor.");
    return nullptr;
  }

  if (QNN_TENSORMEMTYPE_RAW == QNN_TENSOR_GET_MEM_TYPE(tensor)) {
    return QNN_TENSOR_GET_CLIENT_BUF(tensor).data;
  } else if (QNN_TENSORMEMTYPE_MEMHANDLE == QNN_TENSOR_GET_MEM_TYPE(tensor)) {
    if (m_tensorToRpcMem.find(QNN_TENSOR_GET_ID(tensor)) != m_tensorToRpcMem.end()) {
      return m_tensorToRpcMem[QNN_TENSOR_GET_ID(tensor)];
    } else {
      QNN_WARN("getBuffer: Tensor not found");
      return nullptr;
    }
  } else {
    QNN_WARN("getBuffer: Unsupported memType:%d", (int)QNN_TENSOR_GET_MEM_TYPE(tensor));
    return nullptr;
  }
}

// Helper method to read data from files to a buffer.
iotensor::PopulateInputTensorsRetType_t iotensor::IOTensor::readDataAndAllocateBuffer(
    const std::vector<std::string>& filePaths,
    const size_t filePathsIndexOffset,
    const bool loopBackToStart,
    std::vector<size_t> dims,
    Qnn_DataType_t dataType,
    uint8_t** bufferToCopy) {
  StatusCode returnStatus  = StatusCode::SUCCESS;
  *bufferToCopy            = nullptr;
  returnStatus             = allocateBuffer(bufferToCopy, dims, dataType);
  size_t numFilesPopulated = 0;
  size_t batchSize         = 0;
  datautil::StatusCode status;
  std::tie(status, numFilesPopulated, batchSize) =
      datautil::readBatchData(filePaths,
                              filePathsIndexOffset,
                              loopBackToStart,
                              dims,
                              dataType,
                              reinterpret_cast<uint8_t*>(*bufferToCopy));
  if (datautil::StatusCode::SUCCESS != status) {
    QNN_ERROR("Failure in datautil::readBatchData");
    returnStatus = StatusCode::FAILURE;
  }
  if (StatusCode::SUCCESS != returnStatus) {
    if (nullptr != *bufferToCopy) {
      free(*bufferToCopy);
      *bufferToCopy = nullptr;
    }
  }
  return {returnStatus, numFilesPopulated, batchSize};
}

// Helper method to copy a float buffer, quantize it, and copy
// it to a tensor (Qnn_Tensor_t) buffer.
iotensor::StatusCode iotensor::IOTensor::copyFromFloatToNative(float* floatBuffer,
                                                               Qnn_Tensor_t* tensor) {
  if (nullptr == floatBuffer || nullptr == tensor) {
    QNN_ERROR("copyFromFloatToNative(): received a nullptr");
    return StatusCode::FAILURE;
  }

  StatusCode returnStatus = StatusCode::SUCCESS;
  std::vector<size_t> dims;
  fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(tensor), QNN_TENSOR_GET_RANK(tensor));

  switch (QNN_TENSOR_GET_DATA_TYPE(tensor)) {
    case QNN_DATATYPE_UFIXED_POINT_8:
      datautil::floatToTfN<uint8_t>(static_cast<uint8_t*>(getTensorBuffer(tensor)),
                                    floatBuffer,
                                    QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.offset,
                                    QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.scale,
                                    datautil::calculateElementCount(dims));
      break;

    case QNN_DATATYPE_UFIXED_POINT_16:
      datautil::floatToTfN<uint16_t>(static_cast<uint16_t*>(getTensorBuffer(tensor)),
                                     floatBuffer,
                                     QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.offset,
                                     QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.scale,
                                     datautil::calculateElementCount(dims));
      break;

    case QNN_DATATYPE_UINT_8:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<uint8_t>(static_cast<uint8_t*>(getTensorBuffer(tensor)),
                                           floatBuffer,
                                           datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<uint8_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_UINT_16:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<uint16_t>(static_cast<uint16_t*>(getTensorBuffer(tensor)),
                                            floatBuffer,
                                            datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<uint16_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_UINT_32:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<uint32_t>(static_cast<uint32_t*>(getTensorBuffer(tensor)),
                                            floatBuffer,
                                            datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<uint32_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_UINT_64:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<uint64_t>(static_cast<uint64_t*>(getTensorBuffer(tensor)),
                                            floatBuffer,
                                            datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<uint64_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_INT_8:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<int8_t>(static_cast<int8_t*>(getTensorBuffer(tensor)),
                                          floatBuffer,
                                          datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<int8_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_INT_16:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<int16_t>(static_cast<int16_t*>(getTensorBuffer(tensor)),
                                           floatBuffer,
                                           datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<int16_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_INT_32:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<int32_t>(static_cast<int32_t*>(getTensorBuffer(tensor)),
                                           floatBuffer,
                                           datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<int32_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_INT_64:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<int64_t>(static_cast<int64_t*>(getTensorBuffer(tensor)),
                                           floatBuffer,
                                           datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<int64_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_BOOL_8:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castFromFloat<uint8_t>(static_cast<uint8_t*>(getTensorBuffer(tensor)),
                                           floatBuffer,
                                           datautil::calculateElementCount(dims))) {
        QNN_ERROR("failure in castFromFloat<bool>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    default:
      QNN_ERROR("Datatype not supported yet!");
      returnStatus = StatusCode::FAILURE;
      break;
  }
  return returnStatus;
}

// Helper method to populate an input tensor in the graph during execution.
// It relies on reading data from files provided during app creation.
iotensor::PopulateInputTensorsRetType_t iotensor::IOTensor::populateInputTensor(
    const std::vector<std::string>& filePaths,
    const size_t filePathsIndexOffset,
    const bool loopBackToStart,
    Qnn_Tensor_t* input,
    iotensor::InputDataType inputDataType) {
  if (nullptr == input) {
    QNN_ERROR("input is nullptr");
    return {StatusCode::FAILURE, 0, 0};
  }

  auto returnStatus        = StatusCode::SUCCESS;
  size_t numFilesPopulated = 0;
  size_t batchSize         = 0;
  std::vector<size_t> dims;
  fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(input), QNN_TENSOR_GET_RANK(input));

  if (inputDataType == InputDataType::FLOAT &&
      QNN_TENSOR_GET_DATA_TYPE(input) != QNN_DATATYPE_FLOAT_32) {
    uint8_t* fileToBuffer = nullptr;

    std::tie(returnStatus, numFilesPopulated, batchSize) =
        readDataAndAllocateBuffer(filePaths,
                                  filePathsIndexOffset,
                                  loopBackToStart,
                                  dims,
                                  QNN_DATATYPE_FLOAT_32,
                                  &fileToBuffer);
    if (StatusCode::SUCCESS == returnStatus) {
      QNN_DEBUG("readDataFromFileToBuffer successful");
      returnStatus = copyFromFloatToNative(reinterpret_cast<float*>(fileToBuffer), input);
    }
    if (nullptr != fileToBuffer) {
      free(fileToBuffer);
      fileToBuffer = nullptr;
    }
  } else {
    datautil::StatusCode status;
    std::tie(status, numFilesPopulated, batchSize) =
        datautil::readBatchData(filePaths,
                                filePathsIndexOffset,
                                loopBackToStart,
                                dims,
                                QNN_TENSOR_GET_DATA_TYPE(input),
                                reinterpret_cast<uint8_t*>(getTensorBuffer(input)));
    if (datautil::StatusCode::SUCCESS != status) {
      QNN_ERROR("Failure in datautil::readBatchData");
      returnStatus = StatusCode::FAILURE;
    }
  }
  return {returnStatus, numFilesPopulated, batchSize};
}

// Helper method to populate all input tensors during execution.
iotensor::PopulateInputTensorsRetType_t iotensor::IOTensor::populateInputTensors(
    uint32_t graphIdx,
    const std::vector<std::vector<std::string>>& filePathsVector,
    const size_t filePathsIndexOffset,
    const bool loopBackToStart,
    const std::unordered_map<std::string, uint32_t>& inputNameToIndex,
    Qnn_Tensor_t* inputs,
    qnn_wrapper_api::GraphInfo_t graphInfo,
    iotensor::InputDataType inputDataType) {
  if (nullptr == inputs) {
    QNN_ERROR("inputs is nullptr");
    return {StatusCode::FAILURE, 0, 0};
  }
  auto inputCount = graphInfo.numInputTensors;
  if (filePathsVector.size() != inputCount) {
    QNN_ERROR(
        "Incorrect amount of Input files for graphIdx: %d. Expected: %d, "
        "received: %d",
        graphIdx,
        inputCount,
        filePathsVector.size());
    return {StatusCode::FAILURE, 0, 0};
  }
  size_t numFilesPopulated = 0;
  size_t numBatchSize      = 0;
  for (size_t inputIdx = 0; inputIdx < inputCount; inputIdx++) {
    size_t inputNameIdx = inputIdx;
    QNN_DEBUG("index = %d input column index = %d", inputIdx, inputNameIdx);
    std::string inputNodeName;
    if (QNN_TENSOR_GET_NAME(graphInfo.inputTensors[inputIdx]))
      inputNodeName = QNN_TENSOR_GET_NAME(graphInfo.inputTensors[inputIdx]);
    if (!inputNodeName.empty() && inputNameToIndex.find(inputNodeName) != inputNameToIndex.end()) {
      inputNameIdx = inputNameToIndex.at(inputNodeName);
    }
    StatusCode returnStatus;
    size_t currentInputNumFilesPopulated = 0;
    size_t currentInputNumBatchSize      = 0;
    std::tie(returnStatus, currentInputNumFilesPopulated, currentInputNumBatchSize) =
        populateInputTensor(filePathsVector[inputNameIdx],
                            filePathsIndexOffset,
                            loopBackToStart,
                            &(inputs[inputIdx]),
                            inputDataType);
    if (StatusCode::SUCCESS != returnStatus) {
      QNN_ERROR("populateInputTensorFromFiles failed for input %s with index %d",
                inputNodeName.c_str(),
                inputIdx);
      return {StatusCode::FAILURE, currentInputNumFilesPopulated, currentInputNumBatchSize};
    }
    if (inputIdx == 0) {
      numFilesPopulated = currentInputNumFilesPopulated;
      numBatchSize      = currentInputNumBatchSize;
    } else {
      if (numFilesPopulated != currentInputNumFilesPopulated ||
          numBatchSize != currentInputNumBatchSize) {
        QNN_ERROR(
            "Current input tensor with name: %s with index %d files populated = %d, batch size = %d"
            " does not match with expected files populated = %d, batch size = %d",
            inputNodeName.c_str(),
            inputIdx,
            currentInputNumFilesPopulated,
            currentInputNumBatchSize,
            numFilesPopulated,
            numBatchSize);
        return {StatusCode::FAILURE, numFilesPopulated, numBatchSize};
      }
    }
  }
  return {StatusCode::SUCCESS, numFilesPopulated, numBatchSize};
}

// Setup details for Qnn_Tensor_t for execution
// based on information in Qnn_TensorWrapper_t provided by model.so.
iotensor::StatusCode iotensor::IOTensor::setupTensors(Qnn_Tensor_t** tensors,
                                                      uint32_t tensorCount,
                                                      Qnn_Tensor_t* tensorWrappers) {
  if (nullptr == tensorWrappers) {
    QNN_ERROR("tensorWrappers is nullptr");
    return StatusCode::FAILURE;
  }
  if (0 == tensorCount) {
    QNN_INFO("tensor count is 0. Nothing to setup.");
    return StatusCode::SUCCESS;
  }
  auto returnStatus = StatusCode::SUCCESS;
  *tensors          = (Qnn_Tensor_t*)calloc(1, tensorCount * sizeof(Qnn_Tensor_t));
  if (nullptr == *tensors) {
    QNN_ERROR("mem alloc failed for *tensors");
    returnStatus = StatusCode::FAILURE;
    return returnStatus;
  }
  for (size_t tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
    Qnn_Tensor_t wrapperTensor = tensorWrappers[tensorIdx];
    std::vector<size_t> dims;
    fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(wrapperTensor), QNN_TENSOR_GET_RANK(wrapperTensor));
    if (StatusCode::SUCCESS == returnStatus) {
      QNN_DEBUG("allocateBuffer successful");
      (*tensors)[tensorIdx] = QNN_TENSOR_INIT;
      returnStatus =
          (sample_app::deepCopyQnnTensorInfo(((*tensors) + tensorIdx), &wrapperTensor) == true
               ? StatusCode::SUCCESS
               : StatusCode::FAILURE);
    }
    // datautil::StatusCode datautilStatus{datautil::StatusCode::SUCCESS};

    if (m_useSharedBuffer) {
      QNN_DEBUG("Shared buffer mode");
      Qnn_Tensor_t* tensor = (*tensors) + tensorIdx;
      uint8_t* memPointer  = nullptr;
      returnStatus         = allocateBuffer(
          &memPointer, dims, QNN_TENSOR_GET_DATA_TYPE((*tensors) + tensorIdx), m_useSharedBuffer);
      if (nullptr == memPointer) {
        QNN_ERROR("memalloc failed for mem Pointer");
        returnStatus = StatusCode::FAILURE;
        break;
      }

      int memfd = m_rpcMemToFd(memPointer);
      if (-1 == memfd) {
        QNN_ERROR("rpcmem_to_fd failure");
        returnStatus = StatusCode::FAILURE;
        break;
      }

      if (StatusCode::SUCCESS == returnStatus) {
        Qnn_MemDescriptor_t memDescriptor = {
            {QNN_TENSOR_GET_RANK(tensor), QNN_TENSOR_GET_DIMENSIONS(tensor), nullptr},
            QNN_TENSOR_GET_DATA_TYPE(tensor),
            QNN_MEM_TYPE_ION,
            {{-1}}};

        memDescriptor.ionInfo.fd  = memfd;
        Qnn_MemHandle_t memHandle = QNN_TENSOR_GET_MEM_HANDLE(tensor);
        QNN_TENSOR_SET_MEM_TYPE(tensor, QNN_TENSORMEMTYPE_MEMHANDLE);
        QNN_TENSOR_SET_MEM_HANDLE(tensor, memHandle);
        Qnn_ErrorHandle_t returnCode;
        if (QNN_SUCCESS != (returnCode = m_qnnInterface->memRegister(
                                (*m_context), &memDescriptor, 1, &(memHandle)))) {
          const char* errMsg;
          m_qnnInterface->errorGetMessage(returnCode, &errMsg);
          QNN_ERROR("Failure to register ion memory with the backend: %s (%d)", errMsg, returnCode);
          m_qnnInterface->errorGetVerboseMessage(returnCode, &errMsg);
          QNN_ERROR("Verbose error msg: %s", errMsg);
          returnStatus = StatusCode::FAILURE;
        }

        QNN_TENSOR_SET_MEM_HANDLE(tensor, memHandle);
      }

      if (StatusCode::SUCCESS == returnStatus) {
        m_tensorToRpcMem.insert({QNN_TENSOR_GET_ID(tensor), memPointer});
      } else {
        m_rpcMemFree(memPointer);
      }

    } else {
      if (StatusCode::SUCCESS == returnStatus) {
        QNN_TENSOR_SET_MEM_TYPE(((*tensors) + tensorIdx), QNN_TENSORMEMTYPE_RAW);
      }
      Qnn_ClientBuffer_t clientBuffer = QNN_CLIENT_BUFFER_INIT;
      returnStatus = allocateBuffer(reinterpret_cast<uint8_t**>(&clientBuffer.data),
                                    dims,
                                    QNN_TENSOR_GET_DATA_TYPE((*tensors) + tensorIdx),
                                    m_useSharedBuffer);
      datautil::StatusCode datautilStatus{datautil::StatusCode::SUCCESS};
      size_t length{0};
      std::tie(datautilStatus, length) =
          datautil::calculateLength(dims, QNN_TENSOR_GET_DATA_TYPE((*tensors) + tensorIdx));
      if (datautilStatus != datautil::StatusCode::SUCCESS) {
        returnStatus = StatusCode::FAILURE;
      }
      clientBuffer.dataSize = length;
      QNN_TENSOR_SET_CLIENT_BUF(((*tensors) + tensorIdx), clientBuffer);
      if (StatusCode::SUCCESS != returnStatus) {
        QNN_ERROR("Failure in setupTensors, cleaning up resources");
        if (nullptr != (QNN_TENSOR_GET_CLIENT_BUF((*tensors) + tensorIdx)).data) {
          free(QNN_TENSOR_GET_CLIENT_BUF((*tensors) + tensorIdx).data);
        }
        tearDownTensors(*tensors, tensorIdx);
        *tensors     = nullptr;
        returnStatus = StatusCode::FAILURE;
        QNN_ERROR("Failure in setupTensors, done cleaning up resources");
        return returnStatus;
      }
    }
  }
  return returnStatus;
}

// Setup details for all input and output tensors for graph execution.
iotensor::StatusCode iotensor::IOTensor::setupInputAndOutputTensors(
    Qnn_Tensor_t** inputs, Qnn_Tensor_t** outputs, qnn_wrapper_api::GraphInfo_t graphInfo) {
  auto returnStatus = StatusCode::SUCCESS;
  if (StatusCode::SUCCESS !=
      setupTensors(inputs, graphInfo.numInputTensors, (graphInfo.inputTensors))) {
    QNN_ERROR("Failure in setting up input tensors");
    returnStatus = StatusCode::FAILURE;
  }
  if (StatusCode::SUCCESS !=
      setupTensors(outputs, graphInfo.numOutputTensors, (graphInfo.outputTensors))) {
    QNN_ERROR("Failure in setting up output tensors");
    returnStatus = StatusCode::FAILURE;
  }
  if (StatusCode::SUCCESS != returnStatus) {
    QNN_ERROR("Failure in setupInputAndOutputTensors, cleaning up resources");
    if (nullptr != *inputs) {
      QNN_DEBUG("cleaning up input tensors");
      tearDownTensors(*inputs, graphInfo.numInputTensors);
      *inputs = nullptr;
    }
    if (nullptr != *outputs) {
      QNN_DEBUG("cleaning up output tensors");
      tearDownTensors(*outputs, graphInfo.numOutputTensors);
      *outputs = nullptr;
    }
    QNN_ERROR("Failure in setupInputAndOutputTensors, done cleaning up resources");
  }
  return returnStatus;
}

// Clean up all tensors related data after execution.
iotensor::StatusCode iotensor::IOTensor::tearDownTensors(Qnn_Tensor_t* tensors,
                                                         uint32_t tensorCount) {
  for (size_t tensorIdx = 0; tensorIdx < tensorCount; tensorIdx++) {
    Qnn_Tensor_t* tensor = tensors + tensorIdx;
    if (nullptr != QNN_TENSOR_GET_DIMENSIONS(tensors[tensorIdx])) {
      free(QNN_TENSOR_GET_DIMENSIONS(tensors[tensorIdx]));
    }
    if (m_tensorToRpcMem.find(QNN_TENSOR_GET_ID(tensor)) != m_tensorToRpcMem.end()) {
      Qnn_MemHandle_t memHandles = QNN_TENSOR_GET_MEM_HANDLE(tensor);
      if (QNN_SUCCESS != m_qnnInterface->memDeRegister(&memHandles, 1)) {
        QNN_WARN("Failed to deregister shared memory with the backend");
      }
      m_rpcMemFree(m_tensorToRpcMem[QNN_TENSOR_GET_ID(tensor)]);
      m_tensorToRpcMem.erase(QNN_TENSOR_GET_ID(tensor));
    } else if (nullptr != QNN_TENSOR_GET_CLIENT_BUF(tensors[tensorIdx]).data) {
      free(QNN_TENSOR_GET_CLIENT_BUF(tensor).data);
    }
  }
  free(tensors);
  return StatusCode::SUCCESS;
}

// Clean up all input and output tensors after execution.
iotensor::StatusCode iotensor::IOTensor::tearDownInputAndOutputTensors(Qnn_Tensor_t* inputs,
                                                                       Qnn_Tensor_t* outputs,
                                                                       size_t numInputTensors,
                                                                       size_t numOutputTensors) {
  if (nullptr != inputs) {
    QNN_INFO("cleaning up resources for input tensors");
    tearDownTensors(inputs, numInputTensors);
    inputs = nullptr;
  }
  if (nullptr != outputs) {
    QNN_INFO("cleaning up resources for output tensors");
    tearDownTensors(outputs, numOutputTensors);
    outputs = nullptr;
  }
  return StatusCode::SUCCESS;
}

// Helper method to allocate a buffer.
iotensor::StatusCode iotensor::IOTensor::allocateBuffer(uint8_t** buffer,
                                                        std::vector<size_t> dims,
                                                        Qnn_DataType_t dataType,
                                                        bool useSharedBuffer) {
  size_t elementCount = datautil::calculateElementCount(dims);
  auto returnStatus   = StatusCode::SUCCESS;
  switch (dataType) {
    case QNN_DATATYPE_FLOAT_32:
      QNN_DEBUG("allocating float buffer");
      returnStatus =
          allocateBuffer<float>(reinterpret_cast<float**>(buffer), elementCount, useSharedBuffer);
      break;

    case QNN_DATATYPE_UINT_8:
    case QNN_DATATYPE_UFIXED_POINT_8:
      QNN_DEBUG("allocating uint8_t buffer");
      returnStatus = allocateBuffer<uint8_t>(
          reinterpret_cast<uint8_t**>(buffer), elementCount, useSharedBuffer);
      break;

    case QNN_DATATYPE_UINT_16:
    case QNN_DATATYPE_UFIXED_POINT_16:
      QNN_DEBUG("allocating uint16_t buffer");
      returnStatus = allocateBuffer<uint16_t>(
          reinterpret_cast<uint16_t**>(buffer), elementCount, useSharedBuffer);
      break;

    case QNN_DATATYPE_UINT_32:
      QNN_DEBUG("allocating uint32_t buffer");
      returnStatus = allocateBuffer<uint32_t>(
          reinterpret_cast<uint32_t**>(buffer), elementCount, useSharedBuffer);
      break;

    case QNN_DATATYPE_UINT_64:
      QNN_DEBUG("allocating uint64_t buffer");
      returnStatus = allocateBuffer<uint64_t>(
          reinterpret_cast<uint64_t**>(buffer), elementCount, useSharedBuffer);
      break;

    case QNN_DATATYPE_INT_8:
      QNN_DEBUG("allocating int8_t buffer");
      returnStatus =
          allocateBuffer<int8_t>(reinterpret_cast<int8_t**>(buffer), elementCount, useSharedBuffer);
      break;

    case QNN_DATATYPE_INT_16:
      QNN_DEBUG("allocating int16_t buffer");
      returnStatus = allocateBuffer<int16_t>(
          reinterpret_cast<int16_t**>(buffer), elementCount, useSharedBuffer);
      break;

    case QNN_DATATYPE_INT_32:
      QNN_DEBUG("allocating int32_t buffer");
      returnStatus = allocateBuffer<int32_t>(
          reinterpret_cast<int32_t**>(buffer), elementCount, useSharedBuffer);
      break;

    case QNN_DATATYPE_INT_64:
      QNN_DEBUG("allocating int64_t buffer");
      returnStatus = allocateBuffer<int64_t>(
          reinterpret_cast<int64_t**>(buffer), elementCount, useSharedBuffer);
      break;

    case QNN_DATATYPE_BOOL_8:
      QNN_DEBUG("allocating bool buffer");
      returnStatus = allocateBuffer<uint8_t>(
          reinterpret_cast<uint8_t**>(buffer), elementCount, useSharedBuffer);
      break;

    default:
      QNN_ERROR("Datatype not supported yet!");
      returnStatus = StatusCode::FAILURE;
      break;
  }
  return returnStatus;
}

// Helper method to allocate a buffer.
template <typename T>
iotensor::StatusCode iotensor::IOTensor::allocateBuffer(T** buffer,
                                                        size_t& elementCount,
                                                        bool useSharedBuffer) {
  QNN_DEBUG("ElementCount: %d, sizeof(T): %d, total size: %d",
            elementCount,
            sizeof(T),
            elementCount * sizeof(T));

  if (useSharedBuffer) {
    QNN_INFO("Using RPC shared buffer allocation method");
    *buffer =
        (T*)m_rpcMemAlloc(RPCMEM_HEAP_ID_SYSTEM, RPCMEM_DEFAULT_FLAGS, elementCount * sizeof(T));
  } else {
    QNN_INFO("Using normal nonshared allocation method");
    *buffer = (T*)malloc(elementCount * sizeof(T));
  }
  if (nullptr == *buffer) {
    QNN_ERROR("Memory allocation failed.");
    return StatusCode::FAILURE;
  }
  return StatusCode::SUCCESS;
}

// Convert data to float or de-quantization. This is used when
// user requests for float output and the model produces
// non-float output.
#ifndef __hexagon__
iotensor::StatusCode iotensor::IOTensor::convertToFloat(float** out, Qnn_Tensor_t* tensor) {
  if (nullptr == tensor) {
    QNN_ERROR("tensors is nullptr");
    return StatusCode::FAILURE;
  }
  std::vector<size_t> dims;
  fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(tensor), QNN_TENSOR_GET_RANK(tensor));
  auto returnStatus   = StatusCode::SUCCESS;
  size_t elementCount = datautil::calculateElementCount(dims);
  returnStatus        = allocateBuffer<float>(out, elementCount);
  if (StatusCode::SUCCESS != returnStatus) {
    QNN_ERROR("failure in allocateBuffer<float>");
    return returnStatus;
  }
  switch (QNN_TENSOR_GET_DATA_TYPE(tensor)) {
    case QNN_DATATYPE_UFIXED_POINT_8:
      if (datautil::StatusCode::SUCCESS !=
          datautil::tfNToFloat<uint8_t>(
              *out,
              reinterpret_cast<uint8_t*>(getTensorBuffer(tensor)),
              QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.offset,
              QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.scale,
              elementCount)) {
        QNN_ERROR("failure in tfNToFloat<uint8_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_UFIXED_POINT_16:
      if (datautil::StatusCode::SUCCESS !=
          datautil::tfNToFloat<uint16_t>(
              *out,
              reinterpret_cast<uint16_t*>(getTensorBuffer(tensor)),
              QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.offset,
              QNN_TENSOR_GET_QUANT_PARAMS(tensor).scaleOffsetEncoding.scale,
              elementCount)) {
        QNN_ERROR("failure in tfNToFloat<uint8_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_UINT_8:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castToFloat<uint8_t>(
              *out, reinterpret_cast<uint8_t*>(getTensorBuffer(tensor)), elementCount)) {
        QNN_ERROR("failure in castToFloat<uint8_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_UINT_16:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castToFloat<uint16_t>(
              *out, reinterpret_cast<uint16_t*>(getTensorBuffer(tensor)), elementCount)) {
        QNN_ERROR("failure in castToFloat<uint16_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_UINT_32:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castToFloat<uint32_t>(
              *out, reinterpret_cast<uint32_t*>(getTensorBuffer(tensor)), elementCount)) {
        QNN_ERROR("failure in castToFloat<uint32_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_UINT_64:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castToFloat<uint64_t>(
              *out, reinterpret_cast<uint64_t*>(getTensorBuffer(tensor)), elementCount)) {
        QNN_ERROR("failure in castToFloat<uint64_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_INT_8:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castToFloat<int8_t>(
              *out, reinterpret_cast<int8_t*>(getTensorBuffer(tensor)), elementCount)) {
        QNN_ERROR("failure in castToFloat<int8_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_INT_16:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castToFloat<int16_t>(
              *out, reinterpret_cast<int16_t*>(getTensorBuffer(tensor)), elementCount)) {
        QNN_ERROR("failure in castToFloat<int16_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_INT_32:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castToFloat<int32_t>(
              *out, reinterpret_cast<int32_t*>(getTensorBuffer(tensor)), elementCount)) {
        QNN_ERROR("failure in castToFloat<int32_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_INT_64:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castToFloat<int64_t>(
              *out, reinterpret_cast<int64_t*>(getTensorBuffer(tensor)), elementCount)) {
        QNN_ERROR("failure in castToFloat<int64_t>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    case QNN_DATATYPE_BOOL_8:
      if (datautil::StatusCode::SUCCESS !=
          datautil::castToFloat<uint8_t>(
              *out, reinterpret_cast<uint8_t*>(getTensorBuffer(tensor)), elementCount)) {
        QNN_ERROR("failure in castToFloat<bool>");
        returnStatus = StatusCode::FAILURE;
      }
      break;

    default:
      QNN_ERROR("Datatype not supported yet!");
      returnStatus = StatusCode::FAILURE;
      break;
  }
  if (StatusCode::SUCCESS != returnStatus) {
    QNN_DEBUG("freeing *out");
    if (*out != nullptr) {
      free(*out);
      *out = nullptr;
    }
  }
  return returnStatus;
}

// Helper method to convert Output tensors to float and write them
// out to files.
iotensor::StatusCode iotensor::IOTensor::convertAndWriteOutputTensorInFloat(
    Qnn_Tensor_t* output,
    std::vector<std::string> outputPaths,
    std::string fileName,
    size_t outputBatchSize) {
  if (nullptr == output) {
    QNN_ERROR("output is nullptr");
    return StatusCode::FAILURE;
  }

  auto returnStatus = StatusCode::SUCCESS;
  std::vector<size_t> dims;
  fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(output), QNN_TENSOR_GET_RANK(output));
  float* floatBuffer = nullptr;
  returnStatus       = convertToFloat(&floatBuffer, output);
  if (StatusCode::SUCCESS != returnStatus) {
    QNN_ERROR("failure in convertToFloat");
    return StatusCode::FAILURE;
  }
  uint8_t* bufferToWrite = reinterpret_cast<uint8_t*>(floatBuffer);
  if (datautil::StatusCode::SUCCESS !=
      datautil::writeBatchDataToFile(
          outputPaths, fileName, dims, QNN_DATATYPE_FLOAT_32, bufferToWrite, outputBatchSize)) {
    QNN_ERROR("failure in writeBatchDataToFile");
    returnStatus = StatusCode::FAILURE;
  }
  if (nullptr != floatBuffer) {
    QNN_DEBUG("freeing floatBuffer");
    free(floatBuffer);
    floatBuffer = nullptr;
  }
  return returnStatus;
}

// Helper method to write out output. There is no de-quantization here.
// Just write output as is to files.
iotensor::StatusCode iotensor::IOTensor::writeOutputTensor(Qnn_Tensor_t* output,
                                                           std::vector<std::string> outputPaths,
                                                           std::string fileName,
                                                           size_t outputBatchSize) {
  if (nullptr == output) {
    QNN_ERROR("output is nullptr");
    return StatusCode::FAILURE;
  }
  auto returnStatus = StatusCode::SUCCESS;
  std::vector<size_t> dims;
  fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(output), QNN_TENSOR_GET_RANK(output));
  uint8_t* bufferToWrite = reinterpret_cast<uint8_t*>(getTensorBuffer(output));
  if (datautil::StatusCode::SUCCESS !=
      datautil::writeBatchDataToFile(outputPaths,
                                     fileName,
                                     dims,
                                     QNN_TENSOR_GET_DATA_TYPE(output),
                                     bufferToWrite,
                                     outputBatchSize)) {
    QNN_ERROR("failure in writeBatchDataToFile");
    returnStatus = StatusCode::FAILURE;
  }
  return returnStatus;
}

// Write out all output tensors to files. If output_data_type is float,
// then all outputs will be raw floats regardless of what the model outputs.
// If the output_data_type is native, then output is written as produced by the model.
// Also, for native option, a json with quantization parameters is written out.
// If output_data_type is float_and_native, both above are done.
// If the output in the graph is float, then output_data_type has no effect.
iotensor::StatusCode iotensor::IOTensor::writeOutputTensors(uint32_t graphIdx,
                                                            size_t startIdx,
                                                            char* graphName,
                                                            Qnn_Tensor_t* outputs,
                                                            uint32_t numOutputs,
                                                            iotensor::OutputDataType outputDatatype,
                                                            uint32_t graphsCount,
                                                            std::string outputPath,
                                                            size_t numInputFilesPopulated,
                                                            size_t outputBatchSize) {
  if (nullptr == outputs) {
    QNN_ERROR("Received nullptr");
    return StatusCode::FAILURE;
  }
  if (graphsCount > 1) {
    if (nullptr != graphName && strlen(graphName) > 0) {
      outputPath += (pal::Path::getSeparator() + std::string(graphName));
    } else {
      outputPath += (pal::Path::getSeparator() + std::string("Graph_") + std::to_string(graphIdx));
    }
  }
  auto returnStatus = StatusCode::SUCCESS;
  std::vector<std::string> outputPaths;
  for (size_t idx = 0; idx < numInputFilesPopulated; idx++) {
    std::string output = outputPath + (pal::Path::getSeparator() + std::string("Result_") +
                                       std::to_string(graphIdx) + std::to_string(startIdx + idx));
    outputPaths.push_back(output);
  }
  for (size_t outputIdx = 0; outputIdx < numOutputs; outputIdx++) {
    QNN_DEBUG("Writing output for outputIdx: %d", outputIdx);
    std::string outputFilePrefix;
    if (nullptr != QNN_TENSOR_GET_NAME(outputs[outputIdx]) &&
        strlen(QNN_TENSOR_GET_NAME(outputs[outputIdx])) > 0) {
      outputFilePrefix = std::string(QNN_TENSOR_GET_NAME(outputs[outputIdx]));
    } else {
      outputFilePrefix = std::string("Output_") + std::to_string(outputIdx);
    }
    auto outputFile       = outputFilePrefix + std::string(".raw");
    auto outputFileNative = outputFilePrefix + std::string("_native.raw");
    if (QNN_TENSOR_GET_DATA_TYPE(outputs[outputIdx]) == QNN_DATATYPE_FLOAT_32) {
      QNN_DEBUG("Writing in output->dataType == QNN_DATATYPE_FLOAT_32");
      returnStatus =
          writeOutputTensor(&(outputs[outputIdx]), outputPaths, outputFile, outputBatchSize);
    } else if (outputDatatype == OutputDataType::FLOAT_ONLY) {
      QNN_DEBUG("Writing in output->dataType == OutputDataType::FLOAT_ONLY");
      returnStatus = convertAndWriteOutputTensorInFloat(
          &(outputs[outputIdx]), outputPaths, outputFile, outputBatchSize);
    } else if (outputDatatype == OutputDataType::NATIVE_ONLY) {
      QNN_DEBUG("Writing in output->dataType == OutputDataType::NATIVE_ONLY");
      returnStatus =
          writeOutputTensor(&(outputs[outputIdx]), outputPaths, outputFileNative, outputBatchSize);
    } else if (outputDatatype == OutputDataType::FLOAT_AND_NATIVE) {
      QNN_DEBUG("Writing in output->dataType == OutputDataType::FLOAT_AND_NATIVE");
      returnStatus = convertAndWriteOutputTensorInFloat(
          &(outputs[outputIdx]), outputPaths, outputFile, outputBatchSize);
      if (StatusCode::SUCCESS == returnStatus) {
        returnStatus = writeOutputTensor(
            &(outputs[outputIdx]), outputPaths, outputFileNative, outputBatchSize);
      }
    }
  }
  return returnStatus;
}
#endif

// Helper method to allocate a buffer and copy data to it.
iotensor::StatusCode iotensor::IOTensor::allocateAndCopyBuffer(uint8_t** buffer,
                                                               Qnn_Tensor_t* tensor) {
  if (nullptr == tensor) {
    return StatusCode::FAILURE;
  }
  std::vector<size_t> dims;
  fillDims(dims, QNN_TENSOR_GET_DIMENSIONS(tensor), QNN_TENSOR_GET_RANK(tensor));
  datautil::StatusCode datautilStatus;
  size_t length;
  std::tie(datautilStatus, length) =
      datautil::calculateLength(dims, QNN_TENSOR_GET_DATA_TYPE(tensor));
  if (datautilStatus != datautil::StatusCode::SUCCESS) {
    return StatusCode::FAILURE;
  }
  if (StatusCode::SUCCESS != allocateBuffer(buffer, dims, QNN_TENSOR_GET_DATA_TYPE(tensor))) {
    QNN_ERROR("failure in allocateBuffer");
    return StatusCode::FAILURE;
  }
  pal::StringOp::memscpy(
      *buffer, length * sizeof(uint8_t), getTensorBuffer(tensor), length * sizeof(uint8_t));
  return StatusCode::SUCCESS;
}

iotensor::StatusCode iotensor::IOTensor::fillDims(std::vector<size_t>& dims,
                                                  uint32_t* inDimensions,
                                                  uint32_t rank) {
  if (nullptr == inDimensions) {
    QNN_ERROR("input dimensions is nullptr");
    return StatusCode::FAILURE;
  }
  for (size_t r = 0; r < rank; r++) {
    dims.push_back(inDimensions[r]);
  }
  return StatusCode::SUCCESS;
}

iotensor::OutputDataType iotensor::parseOutputDataType(std::string dataTypeString) {
  std::transform(dataTypeString.begin(), dataTypeString.end(), dataTypeString.begin(), ::tolower);
  OutputDataType parsedDataType = OutputDataType::INVALID;
  if (dataTypeString == "float_only") {
    parsedDataType = OutputDataType::FLOAT_ONLY;
  } else if (dataTypeString == "native_only") {
    parsedDataType = OutputDataType::NATIVE_ONLY;
  } else if (dataTypeString == "float_and_native") {
    parsedDataType = OutputDataType::FLOAT_AND_NATIVE;
  }
  return parsedDataType;
}

iotensor::InputDataType iotensor::parseInputDataType(std::string dataTypeString) {
  std::transform(dataTypeString.begin(), dataTypeString.end(), dataTypeString.begin(), ::tolower);
  InputDataType parsedDataType = InputDataType::INVALID;
  if (dataTypeString == "float") {
    parsedDataType = InputDataType::FLOAT;
  } else if (dataTypeString == "native") {
    parsedDataType = InputDataType::NATIVE;
  }
  return parsedDataType;
}
iotensor::IOTensor::~IOTensor() {
  if (m_libCdspRpc) {
    pal::dynamicloading::dlClose(m_libCdspRpc);
  }
}
