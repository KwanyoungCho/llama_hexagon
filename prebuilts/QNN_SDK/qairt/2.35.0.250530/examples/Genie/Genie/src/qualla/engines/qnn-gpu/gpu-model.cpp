//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")

#include <cassert>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

#include "fmt/format.h"
#include "fmt/ranges.h"
#include "fp16/fp16.h"
#include "gpu-model.hpp"
#include "qualla/detail/cache-file.hpp"
#include "qualla/detail/timer.hpp"
#include "qualla/env.hpp"

namespace fs = std::filesystem;

static constexpr uint32_t g_magicNum = 0xC0DE;

#define __ERROR(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_ERROR, fmt::format(__fmt, ##__VA_ARGS__))
#define __WARN(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_WARN, fmt::format(__fmt, ##__VA_ARGS__))
#define __INFO(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_INFO, fmt::format(__fmt, ##__VA_ARGS__))
#define __KPIS(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __DEBUG(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __TRACE(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __KVTRACE(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))

namespace qualla {

QnnGpuModel::QnnGpuModel(Env& env, const Params& params)
    : _modelBaseDir(params.model_basedir), _env(env) {
  // Initialize _qnnApi
  _qnnApi = std::unique_ptr<QnnApi>(new QnnApi());

  _ctxSize    = params.ctx_size;
  _numHeads   = params.num_heads;
  _headDim    = params.head_dim;
  _nVocabSize = params.vocab_size;
#ifdef _WIN32
  _useDmabufIo = false;
#else
  _useDmabufIo = true;
#endif
  // Set up filename list for context binaries.
  for (auto& i : params.model_list) {
    fs::path model_path = _modelBaseDir / fs::path(i);
    if (!fs::is_regular_file(model_path)) {
      __ERROR("Qnn-Gpu-Model : Can't access model file : {}", model_path.string());
      throw std::runtime_error("Qnn-Gpu-Model : Can't access model file : " + model_path.string());
    }
    _modelList.push_back(model_path.string());
  }
}

QnnGpuModel::~QnnGpuModel() { __INFO("Qnn-Gpu-Model : model destruct complete"); }

// Given a filename, initializeModel load and initializes QNN runtime libraries and the model
bool QnnGpuModel::initializeModel(void) {
  qualla::Timer start;

  __INFO("Qnn-Gpu-Model : Model Init Start");

  const std::string backend = "libQnnGpu.so";

  __INFO("Backend Library : {}", backend);
  __INFO("Model Files : {}", _modelList);

  auto logger       = _env.logger();
  uint32_t logLevel = 1;  // error
  std::function<void(const char* fmt, uint32_t level, uint64_t timestamp, va_list args)>
      logCallback = nullptr;
  if (logger) {
    logLevel                          = static_cast<uint32_t>(logger->getMaxLevel());
    GenieLog_Callback_t localCallback = logger->getCallback();
    GenieLog_Handle_t localHandle     = logger->getHandle();
    logCallback                       = [localCallback, localHandle](
                      const char* fmt, uint32_t level, uint64_t timestamp, va_list args) {
      // Convert the parameters to match the GenieLog_Callback_t signature
      GenieLog_Level_t genieLevel = static_cast<GenieLog_Level_t>(level);
      localCallback(localHandle, fmt, genieLevel, timestamp, args);
    };
  }
  if (!_qnnApi->initializeGpu(
          backend, _modelList, static_cast<bool>(logger), logLevel, logCallback)) {
    __ERROR("Qnn-Api : Initialization Failed!");
    return false;
  }

  // Initialize QNN IO Tensor
  if (_useDmabufIo) {
    _ioTensor =
        std::unique_ptr<IOTensor>(new IOTensor(BufferAlloc::DMABUF, _qnnApi->getQnnInterfaceVer()));
  } else {
    _ioTensor = std::unique_ptr<IOTensor>(
        new IOTensor(BufferAlloc::DEFAULT, _qnnApi->getQnnInterfaceVer()));
  }
  _numGraphs = _qnnApi->getGraphsCount();
  __INFO("Qnn-Gpu-Model : initialized with {} graph(s)", _numGraphs);

  qnn_wrapper_api::GraphInfo_t** graphs_info = _qnnApi->getGraphsInfo();
  for (size_t graphIdx = 0; graphIdx < _numGraphs; graphIdx++) {
    qnn_wrapper_api::GraphInfo_t* const graphInfo = graphs_info[graphIdx];
    char* graphName                               = graphInfo->graphName;
    std::string graphStr                          = std::string(graphName);

    // Determine queryLength from input_ids : Shape [batchSize = 1, querySize]
    uint32_t queryLength = 0;
    for (size_t tensorIdx = 0; tensorIdx < graphInfo->numInputTensors; tensorIdx++) {
      std::string tensorName;
      std::vector<size_t> tensorDims;
      auto& tensor = graphInfo->inputTensors[tensorIdx];
      _qnnApi->getTensorNameAndShape(tensorName, tensorDims, tensor);
      if (tensorName == "input_ids") {
        queryLength = tensorDims[tensorDims.back() - 1];
        break;
      }
    }
    if (queryLength == 0) {
      __INFO("Qnn-Gpu-Model : model with invalid query length found");
      return false;
    }

    __INFO("Qnn-Gpu-Model : Loading Model QueryLen : {}, Idx {}, Name {}",
           queryLength,
           graphIdx,
           graphName);
    _modelVariants.insert({queryLength, {graphIdx, graphName}});
  }
  __INFO("Qnn-Gpu-Model : model init complete: {} usec", start.elapsed_usec());

  return true;
}

bool QnnGpuModel::initializeIOTensorPerGraph(qnn_wrapper_api::GraphInfo_t* const& graphInfo,
                                             std::string& graphName,
                                             std::string sharedGraphName) {
  __DEBUG("Qnn-Gpu-Model : GraphName {}, numInputTensors {} numOutputTensors {}",
          graphName,
          graphInfo->numInputTensors,
          graphInfo->numOutputTensors);
  // Setup Inputs
  {
    std::unordered_map<std::string, size_t> inputTensorsSize;
    std::unordered_map<std::string, Qnn_Tensor_t*> sharedTensorMap;
    for (size_t tensorIdx = 0; tensorIdx < graphInfo->numInputTensors; tensorIdx++) {
      std::string tensorName;
      std::vector<size_t> tensorDims;
      auto& tensor = graphInfo->inputTensors[tensorIdx];
      _qnnApi->getTensorNameAndShape(tensorName, tensorDims, tensor);
      auto dims                    = QnnUtils::Dims(tensorDims);
      inputTensorsSize[tensorName] = dims.getSize();
      __DEBUG("Qnn-Gpu-Model : Input Tensor Info {} {} {} {}",
              tensorIdx,
              tensorName,
              tensorDims,
              inputTensorsSize[tensorName]);
      std::vector<QnnUtils::QuantParam> quantParams;
      if (!_qnnApi->getTensorQuantParams(&tensor, quantParams)) {
        quantParams.emplace_back(0, 0);
      }

      std::shared_ptr<QnnUtils::Tensor> tensorUtil =
          std::shared_ptr<QnnUtils::Tensor>(new (std::nothrow) QnnUtils::Tensor);
      tensorUtil->dims                   = dims;
      tensorUtil->dtype                  = QNN_TENSOR_GET_DATA_TYPE(tensor);
      tensorUtil->quantParam             = quantParams;
      _inputSpecs[graphName][tensorName] = tensorUtil;

      if (!sharedGraphName.empty()) {
        // Reuse created buffer from variant with max Query Length
        sharedTensorMap[tensorName] = _inputSpecs[sharedGraphName][tensorName]->tensor;
      }
    }

    Qnn_Tensor_t* tensor_bank = nullptr;
    std::unordered_map<std::string, void*> tensor_ptr_map;
    if (true != _ioTensor->setupTensorWithSharedBuffers(&tensor_bank,
                                                        tensor_ptr_map,
                                                        graphInfo->numInputTensors,
                                                        graphInfo->inputTensors,
                                                        inputTensorsSize,
                                                        sharedTensorMap)) {
      QNN_ERROR("Qnn-Gpu-Model : Error in setting up Output Tensors for graph %s",
                graphName.c_str());
      return false;
    }

    _inputTensors[graphName] = tensor_bank;
    for (auto& [tensorName, tensor_ptr] : tensor_ptr_map) {
      _inputSpecs[graphName][tensorName]->tensor = (Qnn_Tensor_t*)tensor_ptr;
    }
    __DEBUG("Qnn-Gpu-Model : Input Tensor Allocated for {}", graphName);
  }

  // Setup Outputs
  {
    std::unordered_map<std::string, size_t> outputTensorsSize;
    std::unordered_map<std::string, Qnn_Tensor_t*> sharedTensorMap;
    for (size_t tensorIdx = 0; tensorIdx < graphInfo->numOutputTensors; tensorIdx++) {
      std::string tensorName;
      std::vector<size_t> tensorDims;

      auto& tensor = graphInfo->outputTensors[tensorIdx];
      _qnnApi->getTensorNameAndShape(tensorName, tensorDims, tensor);

      std::vector<QnnUtils::QuantParam> quantParams;
      if (!_qnnApi->getTensorQuantParams(&tensor, quantParams)) {
        quantParams.emplace_back(0, 0);
      }

      auto dims                     = QnnUtils::Dims(tensorDims);
      outputTensorsSize[tensorName] = dims.getAlignedSize();

      __DEBUG("Qnn-Gpu-Model : Output Tensor Info {} {} {} {}",
              tensorIdx,
              tensorName,
              tensorDims,
              outputTensorsSize[tensorName]);
      std::shared_ptr<QnnUtils::Tensor> tensorUtil =
          std::shared_ptr<QnnUtils::Tensor>(new (std::nothrow) QnnUtils::Tensor);
      tensorUtil->dims                    = dims;
      tensorUtil->dtype                   = QNN_TENSOR_GET_DATA_TYPE(tensor);
      tensorUtil->quantParam              = quantParams;
      _outputSpecs[graphName][tensorName] = tensorUtil;

      if (!sharedGraphName.empty()) {
        sharedTensorMap[tensorName] = _outputSpecs[sharedGraphName][tensorName]->tensor;
      } else if (tensorName.starts_with("past_")) {
        std::string tensorInName    = tensorName.substr(0, tensorName.size() - 3) + "in";
        sharedTensorMap[tensorName] = _inputSpecs[graphName][tensorInName]->tensor;

        // Update Gpu _kvCache
        auto [type, layer_id] = parseKVTensorName(tensorName);
        _kvCache.push_back(
            GpuKVCache((type == 1), layer_id, _inputSpecs[graphName][tensorInName].get()));
      }
    }

    Qnn_Tensor_t* tensor_bank = nullptr;
    std::unordered_map<std::string, void*> tensor_ptr_map;
    if (true != _ioTensor->setupTensorWithSharedBuffers(&tensor_bank,
                                                        tensor_ptr_map,
                                                        graphInfo->numOutputTensors,
                                                        graphInfo->outputTensors,
                                                        outputTensorsSize,
                                                        sharedTensorMap)) {
      QNN_ERROR("Qnn-Gpu-Model : Error in setting up Output Tensors for graph %s",
                graphName.c_str());
      return false;
    }

    _outputTensors[graphName] = tensor_bank;
    for (auto& [tensorName, tensor_ptr] : tensor_ptr_map) {
      _outputSpecs[graphName][tensorName]->tensor = (Qnn_Tensor_t*)tensor_ptr;
    }

    __DEBUG("Qnn-Gpu-Model : Output Tensor Allocated {} {}", graphName, _outputTensors.size());
  }
  return true;
}

// Once the model has been loaded, initialize IO Tensors
// _ioTensors is initialized by the context for now
bool QnnGpuModel::initializeIOTensors() {
  qualla::Timer start;

  // For QNN-GPU, we have only one context per model.
  bool status = _ioTensor->initialize(_qnnApi->getContexts().back());
  if (!status) {
    __ERROR("Qnn-Gpu-Model : failure to initialize IOTensor");
    return false;
  }
  // Getting graph info, Hardcoding single graph for now.
  qnn_wrapper_api::GraphInfo_t** const& graphsInfo = _qnnApi->getGraphsInfo();

  // Setup Input Tensor Memory
  // Allocate Memory for Graph with Max Query Length
  auto it                                           = _modelVariants.begin();
  uint32_t maxGraphIdx                              = it->second.first;
  std::string maxGraphName                          = it->second.second;
  qnn_wrapper_api::GraphInfo_t* const& maxGraphInfo = graphsInfo[maxGraphIdx];

  __INFO("Qnn-Gpu-Model : Initialized IO for {} {}", maxGraphIdx, maxGraphName);
  initializeIOTensorPerGraph(maxGraphInfo, maxGraphName, "");

  // Reuse memory for remaining graphs
  it++;
  while (it != _modelVariants.end()) {
    uint32_t graphIdx                              = it->second.first;
    std::string graphName                          = it->second.second;
    qnn_wrapper_api::GraphInfo_t* const& graphInfo = graphsInfo[graphIdx];
    __INFO("Initialized IO for {} {} shared with {} {}",
           graphIdx,
           graphName,
           maxGraphIdx,
           maxGraphName);
    initializeIOTensorPerGraph(graphInfo, graphName, maxGraphName);
    it++;
  }
  return true;
}

bool QnnGpuModel::initializeTensorPointers() {
  for (auto variant : _modelVariants) {
    uint32_t queryLength  = variant.first;
    std::string graphName = variant.second.second;

    auto inputSpec  = _inputSpecs[graphName];
    auto outputSpec = _outputSpecs[graphName];

    t_list[queryLength].inputIds    = inputSpec[INPUT_IDS].get();
    t_list[queryLength].attnMask    = inputSpec[ATTN_MASK].get();
    t_list[queryLength].positionIds = inputSpec[POS_IDS].get();
    t_list[queryLength].logits      = outputSpec[LOGITS].get();
    // expandCausalMask : Prepare attention mask in Engine
    // Expanded Attention Mask Input is added directly to attn scores.
    t_list[queryLength].expandCausalMask = (inputSpec[ATTN_MASK]->dtype == QNN_DATATYPE_FLOAT_16);

    auto status =
        !(t_list[queryLength].inputIds == nullptr || t_list[queryLength].attnMask == nullptr ||
          t_list[queryLength].positionIds == nullptr || t_list[queryLength].logits == nullptr);
    if (!status) {
      __ERROR("Qnn-Gpu-Model : error in setting up named tensor pointers for graph %s.",
              graphName.c_str());
      return false;
    }

    QnnUtils::Tensor* t_inputIds    = t_list[queryLength].inputIds;
    QnnUtils::Tensor* t_attnMask    = t_list[queryLength].attnMask;
    QnnUtils::Tensor* t_positionIds = t_list[queryLength].positionIds;

    // Initialize input tensors
    // 1. input_ids. Fill with 0/PAD_TOKEN if known
    int32_t* inputIdBuffer = static_cast<int32_t*>(getBuffer(t_inputIds));
    uint32_t inputIdSize   = getNumElements(t_inputIds);
    if (inputIdBuffer) {
      if (_useDmabufIo) {
        _ioTensor->beforeWriteToBuffer(t_inputIds->tensor);
      }
      std::fill_n(inputIdBuffer, inputIdSize, 0);
      if (_useDmabufIo) {
        _ioTensor->afterWriteToBuffer(t_inputIds->tensor);
      }
    }

    // 2. attention_mask. Fill with -100 for expandCausalMask, 0 is default.
    uint16_t* attnMaskBuffer = static_cast<uint16_t*>(getBuffer(t_attnMask));
    uint32_t attnMaskSize    = getNumElements(t_attnMask);
    if (attnMaskBuffer) {
      if (_useDmabufIo) {
        _ioTensor->beforeWriteToBuffer(t_attnMask->tensor);
      }
      if (t_list[queryLength].expandCausalMask) {
        std::fill_n(attnMaskBuffer, attnMaskSize, fp16_ieee_from_fp32_value(-100.0));
      }
      if (_useDmabufIo) {
        _ioTensor->afterWriteToBuffer(t_attnMask->tensor);
      }
    }

    // 3. position_ids. Fill with _ctx_len - 1.
    uint32_t* positionIdBuffer    = static_cast<uint32_t*>(getBuffer(t_positionIds));
    uint32_t positionIdBufferSize = getNumElements(t_positionIds);
    if (positionIdBuffer) {
      if (_useDmabufIo) {
        _ioTensor->beforeWriteToBuffer(t_positionIds->tensor);
      }
      std::fill_n(positionIdBuffer, positionIdBufferSize, _ctxSize - 1);
      if (_useDmabufIo) {
        _ioTensor->afterWriteToBuffer(t_positionIds->tensor);
      }
    }
  }
  return true;
}

void QnnGpuModel::prepareCausalMask(uint16_t* attnMaskBuffer, uint32_t currQuerySize) {
  float zeroFloat32      = 0.0;
  float negInfFloat32    = -100.0;
  uint16_t zeroFloat16   = fp16_ieee_from_fp32_value(zeroFloat32);
  uint16_t negInfFloat16 = fp16_ieee_from_fp32_value(negInfFloat32);

  for (uint32_t i = 0; i < currQuerySize; ++i) {
    uint32_t ctxOffset  = i * _ctxSize;
    uint32_t selectSize = _numTokensProcessed + _numCurrentTokensProcessed + i + 1;
    uint32_t rejectSize = _ctxSize - selectSize;
    std::fill_n(attnMaskBuffer + ctxOffset, selectSize, zeroFloat16);
    std::fill_n(attnMaskBuffer + ctxOffset + selectSize, rejectSize, negInfFloat16);
  }
}

void QnnGpuModel::setupInputTensors(uint32_t maxQuerySize,
                                    uint32_t currQuerySize,
                                    const std::vector<int32_t>& tokens,
                                    const std::vector<int32_t>& attention_map) {
  QnnUtils::Tensor* t_inputIds    = t_list[maxQuerySize].inputIds;
  QnnUtils::Tensor* t_attnMask    = t_list[maxQuerySize].attnMask;
  QnnUtils::Tensor* t_positionIds = t_list[maxQuerySize].positionIds;
  bool expandCausalMask           = t_list[maxQuerySize].expandCausalMask;

  // Setup 1. input_ids
  // Index of input tokens in the embedding vocabulary
  int32_t* inputIdBuffer = static_cast<int32_t*>(getBuffer(t_inputIds));
  if (inputIdBuffer) {
    if (_useDmabufIo) {
      _ioTensor->beforeWriteToBuffer(t_inputIds->tensor);
    }
    std::copy(tokens.begin() + _numCurrentTokensProcessed,
              tokens.begin() + _numCurrentTokensProcessed + currQuerySize,
              inputIdBuffer);
    if (_useDmabufIo) {
      _ioTensor->afterWriteToBuffer(t_inputIds->tensor);
    }
  }

  // Setup 2. attention_mask
  // Attention Mask
  if (_useDmabufIo) {
    _ioTensor->beforeWriteToBuffer(t_attnMask->tensor);
  }
  // Causal Attention Mask
  if (attention_map.empty()) {
    if (expandCausalMask) {
      uint16_t* attnMaskBuffer = static_cast<uint16_t*>(getBuffer(t_attnMask));
      prepareCausalMask(attnMaskBuffer, currQuerySize);
    } else {
      int32_t* attnMaskBuffer = static_cast<int32_t*>(getBuffer(t_attnMask));
      uint32_t selectSize     = _numTokensProcessed + _numCurrentTokensProcessed + currQuerySize;
      uint32_t rejectSize     = _ctxSize - selectSize;
      std::fill_n(attnMaskBuffer, selectSize, 1);
      std::fill_n(attnMaskBuffer + selectSize, rejectSize, 0);
    }
  }
  if (_useDmabufIo) {
    _ioTensor->afterWriteToBuffer(t_attnMask->tensor);
  }

  // Setup 3. position_ids
  // Indices of positions of each input tokens in position embeddings.
  uint32_t* positionIdBuffer    = static_cast<uint32_t*>(getBuffer(t_positionIds));
  uint32_t positionIdBufferSize = getNumElements(t_positionIds);
  if (positionIdBuffer) {
    if (_useDmabufIo) {
      _ioTensor->beforeWriteToBuffer(t_positionIds->tensor);
    }
    std::fill_n(positionIdBuffer, positionIdBufferSize, _ctxSize - 1);
    std::iota(positionIdBuffer,
              positionIdBuffer + currQuerySize,
              _numTokensProcessed + _numCurrentTokensProcessed);
    if (_useDmabufIo) {
      _ioTensor->afterWriteToBuffer(t_positionIds->tensor);
    }
  }
}

template <class T1, class T2>
inline bool QnnGpuModel::executeModel(T1& input, T2& output, std::string graphName) {
  bool ret = _qnnApi->graphExecute(input, output, graphName, timeLogs);
  if (ret != true) {
    QNN_ERROR("Qnn-Gpu-Model : Error executing inference: %d for graph %s", ret, graphName.c_str());
    return false;
  }
  QNN_DEBUG("Qnn-Gpu-Model : Execute finished for graph %s", graphName.c_str());
  return true;
}

bool QnnGpuModel::runInferenceHelper(std::string& graphName,
                                     int32_t* wait_time_total,
                                     int32_t* exec_time_total) {
  int32_t exec_time = 0;
  int32_t wait_time = 0;
  auto start_time   = std::chrono::steady_clock::now();
  Qnn_Tensor_t* inputTensors;
  Qnn_Tensor_t* outputTensors;
  try {
    inputTensors  = _inputTensors[graphName];
    outputTensors = _outputTensors[graphName];
  } catch (const std::exception e) {
    __DEBUG("Qnn-Gpu-Model : Could not find tensors %s", graphName.c_str());
    return false;
  }
  bool status = executeModel(inputTensors, outputTensors, graphName);
  if (!status) {
    return false;
  }
  auto end_time = std::chrono::steady_clock::now();
  exec_time += static_cast<int32_t>(
      std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time).count());

  *exec_time_total += exec_time;
  *wait_time_total += wait_time;
  return true;
}

size_t QnnGpuModel::runInference(const std::vector<int32_t>& tokens,
                                 const std::vector<int32_t>& attention_map,
                                 std::vector<float>& logits,
                                 bool logits_all) {
  if (_numTokensProcessed + tokens.size() > _ctxSize) {
    std::string errMsg = "Called inference with more tokens than model supports: ";
    errMsg += std::to_string(tokens.size()) + " vs. " + std::to_string(_ctxSize);
    throw std::runtime_error(errMsg);
  }
  auto start            = std::chrono::steady_clock::now();
  int32_t totalWaitTime = 0;
  int32_t totalExecTime = 0;

  // Select Kernel Variant
  auto variants            = _modelVariants.begin();
  uint32_t selectedVariant = variants->first;
  while ((variants != _modelVariants.end()) && (variants->first >= tokens.size())) {
    selectedVariant = variants->first;
    variants++;
  }

  // Setup inputs for inference
  std::string selectedGraph = _modelVariants[selectedVariant].second;
  uint32_t numIters =
      static_cast<uint32_t>(std::ceil(static_cast<float>(tokens.size()) / selectedVariant));
  _numCurrentTokensProcessed = 0;
  for (size_t i = 0; i < numIters; i++) {
    __DEBUG("Qnn-Gpu-Model : {} of {} iterations", i + 1, numIters);
    uint32_t currQuerySize = (_numCurrentTokensProcessed + selectedVariant) <= tokens.size()
                                 ? selectedVariant
                                 : tokens.size() - _numCurrentTokensProcessed;
    setupInputTensors(selectedVariant, currQuerySize, tokens, attention_map);
    bool status = runInferenceHelper(selectedGraph, &totalWaitTime, &totalExecTime);
    if (!status) {
      return 0;
    }
    _numCurrentTokensProcessed += currQuerySize;
    processLogits(selectedVariant, currQuerySize, logits, logits_all);
  }

  // Update the numProcessTokens to updated with processed tokens.
  _numTokensProcessed += _numCurrentTokensProcessed;
  _numCurrentTokensProcessed = 0;

  auto stop = std::chrono::steady_clock::now();
  timeLogs["Run Inference (cpp) "].first += static_cast<double>(
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count());
  timeLogs["Run Inference (cpp) "].second++;
  QNN_DEBUG("[TIME] Wait[%d] Exec[%d]\n", totalWaitTime, totalExecTime);
  if (!logits_all) {
    return 1;
  }
  return tokens.size();
}

size_t QnnGpuModel::runInference(const std::vector<int32_t>& tokens,
                                 Tensor& logits,
                                 bool logits_all) {
  if (_numTokensProcessed + tokens.size() > _ctxSize) {
    std::string errMsg = "Called inference with more tokens than model supports: ";
    errMsg += std::to_string(tokens.size()) + " vs. " + std::to_string(_ctxSize);
    throw std::runtime_error(errMsg);
  }
  auto start            = std::chrono::steady_clock::now();
  int32_t totalWaitTime = 0;
  int32_t totalExecTime = 0;

  // Select Kernel Variant
  auto variants            = _modelVariants.begin();
  uint32_t selectedVariant = variants->first;
  while ((variants != _modelVariants.end()) && (variants->first >= tokens.size())) {
    selectedVariant = variants->first;
    variants++;
  }

  // Setup inputs for inference
  std::string selectedGraph = _modelVariants[selectedVariant].second;
  uint32_t numIters =
      static_cast<uint32_t>(std::ceil(static_cast<float>(tokens.size()) / selectedVariant));
  _numCurrentTokensProcessed = 0;
  for (size_t i = 0; i < numIters; i++) {
    __DEBUG("Qnn-Gpu-Model : {} of {} iterations", i + 1, numIters);
    uint32_t currQuerySize = (_numCurrentTokensProcessed + selectedVariant) <= tokens.size()
                                 ? selectedVariant
                                 : tokens.size() - _numCurrentTokensProcessed;
    setupInputTensors(selectedVariant, currQuerySize, tokens, {});
    bool status = runInferenceHelper(selectedGraph, &totalWaitTime, &totalExecTime);
    if (!status) {
      return 0;
    }
    _numCurrentTokensProcessed += currQuerySize;
    processLogits(selectedVariant, currQuerySize, logits, logits_all);
  }

  // Update the numProcessTokens to updated with processed tokens.
  _numTokensProcessed += _numCurrentTokensProcessed;
  _numCurrentTokensProcessed = 0;

  auto stop = std::chrono::steady_clock::now();
  timeLogs["Run Inference (cpp) "].first += static_cast<double>(
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count());
  timeLogs["Run Inference (cpp) "].second++;
  QNN_DEBUG("[TIME] Wait[%d] Exec[%d]\n", totalWaitTime, totalExecTime);
  if (!logits_all) {
    return 1;
  }
  return tokens.size();
}

// Parse KV$ Tensor names here - supports past_{key,value}_{layer_idx}[_h0]_{in,out}
std::tuple<int, int> QnnGpuModel::parseKVTensorName(std::string name) {
  if (!name.starts_with("past_")) return {0, 0};

  const bool is_key = name.starts_with("past_key");
  const size_t pos0 = (is_key) ? 9 : 11;  // "past_key_" OR "past_value_"
  const size_t pos1 = name.find('_', pos0);

  int layer_idx = static_cast<int>(std::stoi(name.substr(pos0, pos1 - pos0)));

  return std::make_tuple(is_key ? 1 : 2, layer_idx);
}

size_t QnnGpuModel::loadKVCache(const std::string& load_path) {
  std::ifstream fs(load_path, std::ios::in | std::ios::binary);
  if (fs.fail()) {
    __ERROR("Qnn-Gpu-Model : loadKVCache errror reading file {}", load_path);
    return 0;
  }

  CacheFileSpec spec;
  fs.read((char*)&spec, sizeof(spec));
  if (spec.magic != g_magicNum) {
    __ERROR("Qnn-Gpu-Model : loadKVCache expected {} found {:#x}", g_magicNum, spec.magic);
    return 0;
  }

  __INFO(
      "Qnn-Gpu-Model : loadKVCache {{ num_tensors {}, magic {}, dtype {}, n_heads {}, embed_dim {} "
      "update_size {} }}",
      spec.num_tensors,
      spec.magic,
      int(spec.dtype),
      spec.n_heads,
      spec.embed_dim,
      spec.update_size);

  uint32_t maxQuerySize           = _modelVariants.begin()->first;
  QnnUtils::Tensor* t_inputIds    = t_list[maxQuerySize].inputIds;
  QnnUtils::Tensor* t_attnMask    = t_list[maxQuerySize].attnMask;
  QnnUtils::Tensor* t_positionIds = t_list[maxQuerySize].positionIds;

  _numTokensProcessed = static_cast<size_t>(spec.update_size);
  if (_numTokensProcessed > 0) {
    // Loop over _kvCache tensor and read from file
    for (auto cache : _kvCache) {
      if (_useDmabufIo) {
        _ioTensor->beforeWriteToBuffer(t_inputIds->tensor);
      }
      char* buffer = (char*)getBuffer(cache.tensorUtil);
      if (cache.isKey) {
        // Key Cache Dims [1, num_heads, head_dim, ctx_size]
        // float16 bits equivalent to uint16_t
        const size_t copySize = _numTokensProcessed;
        const size_t skipSize = _ctxSize;
        for (int i = 0; i < _numHeads; i++) {
          for (int j = 0; j < _headDim; j++) {
            fs.read(buffer, copySize * sizeof(uint16_t));
            buffer += skipSize * sizeof(uint16_t);
          }
        }
      } else {
        // Value Cache Dims [1, num_heads, ctx_size, head_dim]
        // float16 bits equivalent to uint16_t
        const size_t copySize = _numTokensProcessed * _headDim;
        const size_t skipSize = _ctxSize * _headDim;
        for (int i = 0; i < _numHeads; i++) {
          fs.read(buffer, copySize * sizeof(uint16_t));
          buffer += skipSize * sizeof(uint16_t);
        }
      }
      if (_useDmabufIo) {
        _ioTensor->afterWriteToBuffer(t_inputIds->tensor);
      }
    }
  }
  return _numTokensProcessed;
}

bool QnnGpuModel::saveKVCache(const std::string& save_path) {
  std::ofstream fs(save_path, std::ios::out | std::ios::binary);
  if (fs.fail()) {
    __ERROR("Qnn-Gpu-Model : saveKVCache error opening file : {}", save_path);
    throw std::runtime_error("Failed to write to cache file. Please re-check path");
  }

  const CacheFileSpec::DataType dtype = CacheFileSpec::DataType::FLOAT16_T;

  uint32_t numKVTensors = _kvCache.size();

  // Save the cache file metadata
  CacheFileSpec file_spec(
      numKVTensors, g_magicNum, dtype, 0x0, _numHeads, _headDim, _numTokensProcessed);
  fs.write((char*)&file_spec, sizeof(file_spec));

  __INFO(
      "Qnn-Gpu-Model : saveKVCache {{ num_tensors {}, magic {}, dtype {}, n_heads {}, embed_dim {} "
      "update_size {} }}",
      numKVTensors,
      g_magicNum,
      int(dtype),
      _numHeads,
      _headDim,
      _numTokensProcessed);

  uint32_t maxQuerySize           = _modelVariants.begin()->first;
  QnnUtils::Tensor* t_inputIds    = t_list[maxQuerySize].inputIds;
  QnnUtils::Tensor* t_attnMask    = t_list[maxQuerySize].attnMask;
  QnnUtils::Tensor* t_positionIds = t_list[maxQuerySize].positionIds;

  if (_numTokensProcessed > 0) {
    // Loop over _kvCache tensor and write to file
    for (auto cache : _kvCache) {
      if (_useDmabufIo) {
        _ioTensor->beforeReadFromBuffer(t_inputIds->tensor);
      }
      char* buffer = (char*)getBuffer(cache.tensorUtil);
      if (cache.isKey) {
        // Key Cache Dims [1, num_heads, head_dim, ctx_size]
        // float16 bits equivalent to uint16_t
        const size_t copySize = _numTokensProcessed;
        const size_t skipSize = _ctxSize;
        for (int i = 0; i < _numHeads; i++) {
          for (int j = 0; j < _headDim; j++) {
            fs.write((char*)buffer, copySize * sizeof(uint16_t));
            buffer += skipSize * sizeof(uint16_t);
          }
        }
      } else {
        // Value Cache Dims [1, num_heads, ctx_size, head_dim]
        // float16 bits equivalent to uint16_t
        const size_t copySize = _numTokensProcessed * _headDim;
        const size_t skipSize = _ctxSize * _headDim;
        for (int i = 0; i < _numHeads; i++) {
          fs.write((char*)buffer, copySize * sizeof(uint16_t));
          buffer += skipSize;
        }
      }
      if (_useDmabufIo) {
        _ioTensor->afterReadFromBuffer(t_inputIds->tensor);
      }
    }
  }
  fs.flush();
  fs.close();

  return true;
}

size_t QnnGpuModel::processLogits(std::uint32_t graphVariant,
                                  uint32_t currQuerySize,
                                  std::vector<float>& logits,
                                  bool logits_all) {
  if (!logits_all) {
    logits.clear();
  }
  size_t logitsTensorSize =
      logits_all ? (logits.size() + currQuerySize * _nVocabSize) : _nVocabSize;
  logits.reserve(logitsTensorSize);

  auto logitsTensor = t_list[graphVariant].logits;
  if (_useDmabufIo) {
    _ioTensor->beforeReadFromBuffer(logitsTensor->tensor);
  }
  uint16_t* logitBuf = static_cast<uint16_t*>(getBuffer(logitsTensor));

  if (!logits_all) {
    logitBuf += (currQuerySize - 1) * _nVocabSize;
  }
  for (auto i = 0; i < logitsTensorSize; ++i) {
    logits.push_back(fp16_ieee_to_fp32_value(logitBuf[i]));
  }
  if (_useDmabufIo) {
    _ioTensor->afterReadFromBuffer(logitsTensor->tensor);
  }

  return logitsTensorSize / _nVocabSize;
}

// GPU only outputs in float16
// will be using Tensor class owned memeory for this
size_t QnnGpuModel::processLogits(std::uint32_t graphVariant,
                                  uint32_t currQuerySize,
                                  Tensor& logits,
                                  bool logits_all) {
  if (!logits_all) {
    logits.logits.clear();
  }
  size_t logitsTensorSize =
      logits_all ? (logits.logits.size() + currQuerySize * _nVocabSize) : _nVocabSize;
  logits.setSize(logitsTensorSize);
  logits.logits.reserve(logitsTensorSize);

  auto logitsTensor = t_list[graphVariant].logits;
  if (_useDmabufIo) {
    _ioTensor->beforeReadFromBuffer(logitsTensor->tensor);
  }
  uint16_t* logitBuf = static_cast<uint16_t*>(getBuffer(logitsTensor));
  if (!logits_all) {
    logitBuf += (currQuerySize - 1) * _nVocabSize;
  }

  for (auto i = 0; i < logitsTensorSize; ++i) {
    logits.logits.push_back(fp16_ieee_to_fp32_value(logitBuf[i]));
  }
  if (_useDmabufIo) {
    _ioTensor->afterReadFromBuffer(logitsTensor->tensor);
  }
  logits.setData((void*)(logits.logits.data()));
  logits.setDataType(TENSOR_DATATYPE_FLOAT_32);

  return logits.getSize() / _nVocabSize;
}

bool QnnGpuModel::reset() {
  // Reset Token Counter
  _numTokensProcessed = 0;

  uint32_t maxQuerySize           = _modelVariants.begin()->first;
  QnnUtils::Tensor* t_inputIds    = t_list[maxQuerySize].inputIds;
  QnnUtils::Tensor* t_attnMask    = t_list[maxQuerySize].attnMask;
  QnnUtils::Tensor* t_positionIds = t_list[maxQuerySize].positionIds;
  bool expandCausalMask           = t_list[maxQuerySize].expandCausalMask;

  // Reset Attention Mask
  uint32_t* attnMaskBuffer = (uint32_t*)getBuffer(t_attnMask);
  uint32_t attnMaskSize    = getNumElements(t_attnMask);
  if (attnMaskBuffer) {
    if (_useDmabufIo) {
      _ioTensor->beforeWriteToBuffer(t_attnMask->tensor);
    }
    if (expandCausalMask) {
      std::fill_n(attnMaskBuffer, attnMaskSize, fp16_ieee_from_fp32_value(-100.0));
    } else {
      std::fill_n(attnMaskBuffer, attnMaskSize, 0);
    }
    if (_useDmabufIo) {
      _ioTensor->afterWriteToBuffer(t_attnMask->tensor);
    }
  }

  // Reset KV Cache.
  // TODO : Check if mask_neg -100 is enough to remove
  // effect of KV Cache. Test with mask_neg = -float_inf
  for (auto cache : _kvCache) {
    if (_useDmabufIo) {
      _ioTensor->beforeWriteToBuffer(t_inputIds->tensor);
    }
    char* buffer        = (char*)getBuffer(cache.tensorUtil);
    uint32_t bufferSize = getBufferSize(cache.tensorUtil);
    memset(buffer, 0, bufferSize);
    if (_useDmabufIo) {
      _ioTensor->afterWriteToBuffer(t_inputIds->tensor);
    }
  }
  return true;
}

}  // namespace qualla
