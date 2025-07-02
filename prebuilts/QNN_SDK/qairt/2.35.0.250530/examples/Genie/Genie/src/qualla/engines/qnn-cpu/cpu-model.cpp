//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <cassert>
#include <cstring>
#include <fstream>
#include <set>
#include <sstream>

#include "cpu-model.hpp"
#include "fmt/format.h"
#include "fmt/ranges.h"
#include "qnn-utils.hpp"
#include "qualla/detail/cache-file.hpp"
#include "qualla/detail/timer.hpp"
#include "qualla/env.hpp"

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

namespace fs = std::filesystem;

namespace qualla {

QnnCpuModel::QnnCpuModel(Env& env, const Params& params)
    : _env(env),
      model_basedir(params.model_basedir),
      op_package(params.op_package),
      backend_lib(params.backend_lib),
      model_bin_path(params.model_bin_path),
      model(params.model),
      model_input(params.model_input),
      model_output(params.model_output),
      embedding_datatype(params.embedding_datatype),
      m_ctx_size(params.ctx_size),
      m_num_layer(params.n_layer),
      m_embd(params.n_embd),
      m_num_heads(params.n_heads),
      m_num_kv_heads(params.n_kv_heads),
      m_num_tokens(params.ctx_size),
      m_num_threads(params.n_threads),
      m_numLogits(params.n_logits),
      m_vocab_size(params.n_vocab_size),
      m_use_mmap(params.use_mmap),
      m_kv_quant(params.kv_quant) {
  // Initialize QnnAPI
  m_qnnApi   = std::unique_ptr<QnnApi>(new QnnApi());
  m_head_dim = m_embd / m_num_heads;
  m_kv_scale_dim.push_back(m_num_layer);
  m_kv_scale_dim.push_back(m_num_kv_heads);
  m_kv_scale_dim.push_back(m_ctx_size + 1);
  m_kv_scale_dim.push_back(m_head_dim / 32);
  // K$, V$ 4D Tensor {n_layer, n_kv_heads, n_ctx, n_head_dim}
  m_kv_dim.push_back(m_num_layer);
  m_kv_dim.push_back(m_num_kv_heads);
  m_kv_dim.push_back(m_ctx_size + 1);
  m_kv_dim.push_back(m_head_dim);
  if (model_input == ModelInput::TOKENS) {
    m_input_dim.push_back(1);
    m_input_dim.push_back(m_ctx_size);
  } else if (model_input == ModelInput::INPUT_EMBEDDINGS) {
    m_input_dim.push_back(m_ctx_size);
    m_input_dim.push_back(m_embd);
  }
  if (model_output == ModelOutput::LOGITS) {
    m_output_dim.push_back(m_numLogits);
    m_output_dim.push_back(m_vocab_size);
  } else if (model_output == ModelOutput::EMBEDDINGS) {
    m_numLogits = m_ctx_size;
    m_output_dim.push_back(m_numLogits);
    m_output_dim.push_back(m_embd);
  }
  if (embedding_datatype == "QNN_DATATYPE_FLOAT_32") {
    m_embeddingBufferSize = m_embd * sizeof(float);
  }
  m_loraConfigType = params.lora_config_type;
  if (m_loraConfigType == LoraConfigType::LORA_ADAPTER_WEIGHT_ENABLE) {
    m_loraConfig.insert(params.lora_config.begin(), params.lora_config.end());
  }
  m_lora_alpha_val = {};
  for (auto it : m_loraConfig) {
    for (auto idx = 0; idx < it.second.alphas.size(); idx++) {
      m_lora_alpha_val[it.second.alphas[idx]] = it.second.alpha_tensor_val[idx];
    }
  }
}

QnnCpuModel::~QnnCpuModel() {
  // Free Qnn Tensor and their memory
  if (dequant_logits_ptr != nullptr) free(dequant_logits_ptr);
  if (m_ioTensor) {
    QNN_DEBUG("Tearing Down Input Tensors Bank");
    for (auto& graph_name : model_order) {
      m_ioTensor->tearDownTensors(m_input_tensors[graph_name], m_input_specs[graph_name].size());
      m_ioTensor->tearDownTensors(m_output_tensors[graph_name], m_output_specs[graph_name].size());
    }
  }
}

// Given a filename, initializeModel load and initializes QNN runtime libraries and the model
bool QnnCpuModel::initializeModel(void) {
  // prepare params
  Qnn_Param_t params[7];
  params[0].paramType               = QNN_PARAMTYPE_SCALAR;
  params[0].name                    = (char*)("model_bin_path");
  params[0].scalarParam.dataType    = QNN_DATATYPE_STRING;
  params[0].scalarParam.stringValue = model_bin_path.c_str();

  params[1].paramType               = QNN_PARAMTYPE_SCALAR;
  params[1].name                    = (char*)("num_thread");
  params[1].scalarParam.dataType    = QNN_DATATYPE_UINT_32;
  params[1].scalarParam.uint32Value = m_num_threads;

  params[2].paramType               = QNN_PARAMTYPE_SCALAR;
  params[2].name                    = (char*)("num_context");
  params[2].scalarParam.dataType    = QNN_DATATYPE_UINT_32;
  params[2].scalarParam.uint32Value = m_ctx_size;

  params[3].paramType               = QNN_PARAMTYPE_SCALAR;
  params[3].name                    = (char*)("num_last_logits");
  params[3].scalarParam.dataType    = QNN_DATATYPE_UINT_32;
  params[3].scalarParam.uint32Value = m_numLogits;

  params[4].paramType               = QNN_PARAMTYPE_SCALAR;
  params[4].name                    = (char*)("use_mmap");
  params[4].scalarParam.dataType    = QNN_DATATYPE_BOOL_8;
  params[4].scalarParam.uint32Value = m_use_mmap;

  params[5].paramType               = QNN_PARAMTYPE_SCALAR;
  params[5].name                    = (char*)("kv_quant");
  params[5].scalarParam.dataType    = QNN_DATATYPE_BOOL_8;
  params[5].scalarParam.uint32Value = m_kv_quant;

  params[6].paramType               = QNN_PARAMTYPE_SCALAR;
  params[6].name                    = (char*)("input_type");
  params[6].scalarParam.dataType    = QNN_DATATYPE_UINT_32;
  params[6].scalarParam.uint32Value = model_input;

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
  if (true != m_qnnApi->initializeCpu(backend_lib,
                                      model,
                                      op_package,
                                      {},
                                      m_input_dim.data(),
                                      m_input_dim.size(),
                                      m_output_dim.data(),
                                      m_output_dim.size(),
                                      m_kv_dim.data(),
                                      m_kv_dim.size(),
                                      m_kv_scale_dim.data(),
                                      params,
                                      7,
                                      false,
                                      static_cast<bool>(logger),
                                      logLevel,
                                      logCallback)) {
    QNN_ERROR("Backend library : %s", backend_lib.c_str());
    throw std::runtime_error("QNN initialization failed!");
  }

  // Initialize QNN IO Tensor
  m_ioTensor   = std::unique_ptr<IOTensor>(new IOTensor());
  m_num_graphs = m_qnnApi->getGraphsCount();
  QNN_DEBUG("QNN initialized with %u graph(s)", m_num_graphs);

  auto graphs_info = m_qnnApi->getGraphsInfo();
  for (size_t graph_idx = 0; graph_idx < m_num_graphs; graph_idx++) {
    qnn_wrapper_api::GraphInfo_t* const& graph_info = graphs_info[graph_idx];
    char* graph_name                                = graph_info->graphName;
    std::string graph_str                           = std::string(graph_name);

    QNN_DEBUG("Loaded graph[%lu] = %s", graph_idx, graph_name);
    model_order.push_back(graph_str);
    model_context[graph_str] =
        m_qnnApi->getContexts()[graph_idx / m_qnnApi->getGraphCountPerContext()];
  }

  // CPU support KV cache mode
  m_mode = ExecutionMode::KV_ONLY;

  return true;
}

// Once the model has been loaded, initialize IO Tensors
// m_ioTensors is initialized by the context for now
bool QnnCpuModel::initializeIOTensors() {
  QNN_DEBUG("Create input tensors bank");

  // Ideally, we should create and initalize m_ioTensor for each context, but we want to
  // be able to see/use all the buffers in every contexts so that they can be connected
  // with each other. Hence, we are using only the first context to initialize the m_ioTensor
  // and use it for all graphs/contexts.
  if (true != m_ioTensor->initialize(m_qnnApi->getContexts()[0])) {
    QNN_ERROR("Failure to initialize IOTensor");
    return false;
  }

  // Getting graph info and its count needed for subsequent steps
  qnn_wrapper_api::GraphInfo_t** const& graphsInfo = m_qnnApi->getGraphsInfo();

  for (size_t graphIdx = 0; graphIdx < m_num_graphs; graphIdx++) {
    qnn_wrapper_api::GraphInfo_t* const& graphInfo = graphsInfo[graphIdx];
    std::string graphName                          = std::string(graphInfo->graphName);

    // Setup Inputs
    {
      std::unordered_map<std::string, size_t> inputTensorsSize;
      for (size_t tensorIdx = 0; tensorIdx < graphInfo->numInputTensors; tensorIdx++) {
        std::string tensor_name;
        std::vector<size_t> tensorDims;

        auto& tensor = graphInfo->inputTensors[tensorIdx];
        m_qnnApi->getTensorNameAndShape(tensor_name, tensorDims, tensor);
        std::vector<QnnUtils::QuantParam> quantParams;
        if (!m_qnnApi->getTensorQuantParams(&tensor, quantParams)) {
          QNN_DEBUG("Couldn't get tensor quant params : %s", tensor_name.c_str());
          quantParams.emplace_back(0, 0);
        }

        auto dims                     = QnnUtils::Dims(tensorDims);
        inputTensorsSize[tensor_name] = dims.getAlignedSize();

        m_input_specs[graphName][tensor_name] = {&tensor};
      }

      Qnn_Tensor_t* tensor_bank = nullptr;
      std::unordered_map<std::string, void*> tensor_ptr_map;
      if (true != m_ioTensor->setupInputTensors(&tensor_bank,
                                                tensor_ptr_map,
                                                *graphInfo,
                                                inputTensorsSize,
                                                m_qnnApi->getContexts()[graphIdx],
                                                false)) {
        QNN_ERROR("Error in setting up Input Tensors for graph %s", graphName.c_str());
        return false;
      }

      m_input_tensors[graphName] = tensor_bank;
      for (auto& [tensor_name, tensor_ptr] : tensor_ptr_map) {
        m_input_specs[graphName][tensor_name].tensor = (Qnn_Tensor_t*)tensor_ptr;
      }
    }

    // Setup Outputs
    {
      std::unordered_map<std::string, size_t> outputTensorsSize;
      for (size_t tensorIdx = 0; tensorIdx < graphInfo->numOutputTensors; tensorIdx++) {
        std::string tensor_name;
        std::vector<size_t> tensorDims;

        auto& tensor = graphInfo->outputTensors[tensorIdx];
        m_qnnApi->getTensorNameAndShape(tensor_name, tensorDims, tensor);
        std::vector<QnnUtils::QuantParam> quantParams;
        if (!m_qnnApi->getTensorQuantParams(&tensor, quantParams)) {
          QNN_DEBUG("Couldn't get tensor quant params : %s", tensor_name.c_str());
          quantParams.emplace_back(0, 0);
        }

        auto dims                      = QnnUtils::Dims(tensorDims);
        outputTensorsSize[tensor_name] = dims.getAlignedSize();

        m_output_specs[graphName][tensor_name] = {&tensor};
      }

      Qnn_Tensor_t* tensor_bank = nullptr;
      std::unordered_map<std::string, void*> tensor_ptr_map;
      if (true != m_ioTensor->setupOutputTensors(&tensor_bank,
                                                 tensor_ptr_map,
                                                 *graphInfo,
                                                 outputTensorsSize,
                                                 m_qnnApi->getContexts()[graphIdx],
                                                 false)) {
        QNN_ERROR("Error in setting up Output Tensors for graph %s", graphName.c_str());
        return false;
      }

      m_output_tensors[graphName] = tensor_bank;
      for (auto& [tensor_name, tensor_ptr] : tensor_ptr_map) {
        m_output_specs[graphName][tensor_name].tensor = (Qnn_Tensor_t*)tensor_ptr;
      }
    }
  }

#ifdef DUMP_TENSOR_SPECS
  dumpTensorSpecs();
#endif

  return true;
}

void QnnCpuModel::dumpTensorSpecs() {
#ifdef DEBUG_DUMP_TARGET_PATH
  if (true != QnnUtils::CreateDirsIfNotExist(DEBUG_DUMP_TARGET_PATH)) {
    throw std::runtime_error(std::string("Could not create directory : ") + DEBUG_DUMP_TARGET_PATH);
  }

  static const char* stringFmt =
      "\t\t{ \"name\": \"%s\", \"dims\": [1, %d, %d, %d], \"bitwidth\": %d, \"scale\": [%s], "
      "\"offset\": [%s] },\n";

  GraphInfo_t** const& graphsInfo = m_qnnApi->getGraphsInfo();
  for (size_t graphIdx = 0; graphIdx < m_num_graphs; graphIdx++) {
    GraphInfo_t* const& graphInfo = graphsInfo[graphIdx];
    std::string graphName         = std::string(graphInfo->graphName);

    // Create output spec file and open it
    char filename[255];
    sprintf(filename, "%s/spec.%s.json", DEBUG_DUMP_TARGET_PATH, graphInfo->graphName);

    FILE* specFile = fopen(filename, "w");
    if (specFile == NULL) {
      throw std::runtime_error(std::string("Error opening file : ") + filename);
    }

    fprintf(specFile, "{\n\t\"graph_name\" : \"%s\",\n\t\"inputs\" : [\n", graphName.c_str());

    std::string tensor_name;
    std::vector<size_t> tensorDims;

    for (size_t tensorIdx = 0; tensorIdx < graphInfo->numInputTensors; tensorIdx++) {
      auto& tensor = graphInfo->inputTensors[tensorIdx];
      m_qnnApi->getTensorNameAndShape(tensor_name, tensorDims, tensor);
      std::string fixed_tensor_name = tensor_name.substr(0, tensor_name.find("_converted"));
      QnnUtils::Tensor& spec        = m_input_specs[graphName][fixed_tensor_name];
      std::string scales;
      std::string offsets;
      getQuantParamString(spec.quantParam, scales, offsets);
      fprintf(specFile,
              stringFmt,
              tensor_name.c_str(),
              spec.dims.height,
              spec.dims.width,
              spec.dims.channel,
              spec.dims.bitWidth,
              scales.c_str(),
              offsets.c_str());
    }

    fseek(specFile, -2, SEEK_CUR);  // Remove trailing comma

    // Dump out output tensor specs
    fprintf(specFile, "\n\t],\n\t\"outputs\" : [\n");

    for (size_t tensorIdx = 0; tensorIdx < graphInfo->numOutputTensors; tensorIdx++) {
      auto& tensor = graphInfo->outputTensors[tensorIdx];
      m_qnnApi->getTensorNameAndShape(tensor_name, tensorDims, tensor);
      std::string fixed_tensor_name = tensor_name.substr(0, tensor_name.find("_converted"));
      QnnUtils::Tensor& spec        = m_output_specs[graphName][fixed_tensor_name];
      std::string scales;
      std::string offsets;
      getQuantParamString(spec.quantParam, scales, offsets);
      fprintf(specFile,
              stringFmt,
              tensor_name.c_str(),
              spec.dims.height,
              spec.dims.width,
              spec.dims.channel,
              spec.dims.bitWidth,
              scales.c_str(),
              offsets.c_str());
    }
    fseek(specFile, -2, SEEK_CUR);  // Remove trailing comma
    fprintf(specFile, "\n\t]\n}");

    fclose(specFile);
  }
#else
  QNN_ERROR(
      "Requested dump tensor specs, but DEBUG_DUMP_TARGET_PATH not set. Please check nsp-model.h");
#endif
}

template <bool PrintError = true, typename ValType>
inline bool findTensor(std::unordered_map<std::string, ValType>& map, std::string key) {
  if (map.find(key) == map.end()) {
    if constexpr (PrintError == true) QNN_ERROR("Cannot find %s\n", key.c_str());
    return false;
  }
  return true;
}

template <bool PrintError = false, typename ValType>
inline ValType* getTensor(std::unordered_map<std::string, ValType>& map, std::string key) {
  if (map.find(key) == map.end()) {
    if constexpr (PrintError == true) QNN_ERROR("Cannot find %s\n", key.c_str());
    return nullptr;
  }
  return &map[key];
}

// Run all validations for the model here so we can exit early
bool QnnCpuModel::validateModel() { return true; }

bool QnnCpuModel::initializeTensorPointers() {
  auto& input_specs         = m_input_specs[model_order.back()];
  t_input_ids               = &input_specs["x0"];
  t_input_ids_num_token     = &input_specs["x1"];
  t_input_ids_reset_kvcache = &input_specs["x2"];
  t_input_ids_k_cache       = &input_specs["x3"];
  t_input_ids_v_cache       = &input_specs["x4"];
  t_input_ids_n_past        = &input_specs["x5"];
  t_input_lora_alpha        = &input_specs["x6"];
  if (m_kv_quant) {
    t_input_ids_k_scale = &input_specs["x7"];
    t_input_ids_v_scale = &input_specs["x8"];
  }
  auto& output_specs = m_output_specs[model_order.back()];
  t_logits           = &output_specs["output_genAI"];
  t_output_n_past    = &output_specs["output_npast"];
  return true;
}

void QnnCpuModel::setupInputTensors(const std::vector<int32_t>& tokens, bool run_bert_mode) {
  size_t num_tokens = m_num_tokens;

  if (tokens.size() > num_tokens) {
    std::string err_msg = "Called inference with more tokens than model supports: ";
    err_msg += std::to_string(tokens.size()) + " vs. " + std::to_string(num_tokens);
    throw std::runtime_error(err_msg);
  }

  // Grab pointers to buffers for access
  uint32_t* input_id_buffer               = (uint32_t*)getBuffer(t_input_ids);
  uint32_t* input_id_num_token_buffer     = (uint32_t*)getBuffer(t_input_ids_num_token);
  uint32_t* input_id_reset_kvcache_buffer = (uint32_t*)getBuffer(t_input_ids_reset_kvcache);
  uint32_t* input_id_n_past_buffer        = (uint32_t*)getBuffer(t_input_ids_n_past);
  float* input_id_lora_alpha              = (float*)getBuffer(t_input_lora_alpha);

  uint32_t size = 1;
  for (auto dim : m_input_dim) {
    size *= dim;
  }

  std::memset(input_id_buffer, 0, size * sizeof(uint32_t));
  std::memset(input_id_n_past_buffer, 0, sizeof(uint32_t));
  std::memset(input_id_num_token_buffer, 0, sizeof(uint32_t));
  std::memset(input_id_reset_kvcache_buffer, 0, sizeof(uint32_t));

  std::memcpy(input_id_buffer, tokens.data(), tokens.size() * sizeof(uint32_t));
  *input_id_num_token_buffer = tokens.size();
  *input_id_n_past_buffer    = m_nPast;

  if (m_adapter.empty()) return;
  for (auto idx = 0; idx < m_loraConfig[m_adapter].alphas.size(); idx++) {
    m_loraConfig[m_adapter].alpha_tensor_val[idx] =
        m_lora_alpha_val[m_loraConfig[m_adapter].alphas[idx]];
  }
  std::memcpy(input_id_lora_alpha,
              m_loraConfig[m_adapter].alpha_tensor_val.data(),
              m_loraConfig[m_adapter].alpha_tensor_val.size() * sizeof(uint32_t));
}

void QnnCpuModel::setupInputTensors(std::vector<uint8_t>& embeddings, bool run_bert_mode) {
  size_t num_tokens         = m_num_tokens;
  const size_t embedBufSize = m_embeddingBufferSize;
  int32_t num_input_tokens  = embeddings.size() / embedBufSize;

  if (num_input_tokens > num_tokens) {
    std::string err_msg = "Called inference with more tokens than model supports: ";
    err_msg += "embedding size" + std::to_string(embeddings.size()) + "tokens size" +
               std::to_string(num_input_tokens) + " vs. " + std::to_string(num_tokens);
    throw std::runtime_error(err_msg);
  }

  // Grab pointers to buffers for access
  float* input_id_buffer                  = (float*)getBuffer(t_input_ids);
  uint32_t* input_id_num_token_buffer     = (uint32_t*)getBuffer(t_input_ids_num_token);
  uint32_t* input_id_reset_kvcache_buffer = (uint32_t*)getBuffer(t_input_ids_reset_kvcache);
  uint32_t* input_id_n_past_buffer        = (uint32_t*)getBuffer(t_input_ids_n_past);
  float* input_id_lora_alpha              = (float*)getBuffer(t_input_lora_alpha);

  std::memset(input_id_reset_kvcache_buffer, 0, sizeof(uint32_t));

  std::memcpy(input_id_buffer, embeddings.data(), embeddings.size());
  *input_id_num_token_buffer = num_input_tokens;
  *input_id_n_past_buffer    = m_nPast;

  if (m_adapter.empty()) return;
  for (auto idx = 0; idx < m_loraConfig[m_adapter].alphas.size(); idx++) {
    m_loraConfig[m_adapter].alpha_tensor_val[idx] =
        m_lora_alpha_val[m_loraConfig[m_adapter].alphas[idx]];
  }
  std::memcpy(input_id_lora_alpha,
              m_loraConfig[m_adapter].alpha_tensor_val.data(),
              m_loraConfig[m_adapter].alpha_tensor_val.size() * sizeof(uint32_t));
}

// Use qnnAPI to execute the model
template <class T1, class T2>
inline bool QnnCpuModel::executeModel(T1& input, T2& output, std::string graph_name) {
  // given that a dnn instance is created and we have input loaded with image data we can get our
  // output for our required app functionality Execute the network with the given single input.
  QNN_DEBUG("Now executing inference for graph %s", graph_name.c_str());

#ifdef INPUT_DUMP
  if (m_inference_count < 5) dumpTensors(graph_name, true);  // Dump input tensors
#endif
  bool ret = m_qnnApi->graphExecute(input, output, graph_name, timeLogs);

  if (ret != true) {
    QNN_ERROR("ERROR executing inference: %d for graph %s", ret, graph_name.c_str());
    return false;
  }
#ifdef OUTPUT_DUMP
  if (m_inference_count < 5) dumpTensors(graph_name, false);  // Dump output tensors
#endif
  QNN_DEBUG("Execute finished for graph %s", graph_name.c_str());

  return true;
}

bool QnnCpuModel::runInferenceHelper(std::vector<std::string>& exec_models,
                                     int32_t* wait_time_total,
                                     int32_t* exec_time_total,
                                     bool pipeline_kv_update,
                                     size_t update_size) {
  int32_t exec_time = 0;
  int32_t wait_time = 0;
  for (auto& graph_name : exec_models) {
    {
      auto startTime = std::chrono::steady_clock::now();
      if (true !=
          executeModel(m_input_tensors[graph_name], m_output_tensors[graph_name], graph_name)) {
        return false;
      }
      auto endTime = std::chrono::steady_clock::now();
      exec_time += static_cast<int32_t>(
          std::chrono::duration_cast<std::chrono::microseconds>(endTime - startTime).count());
    }
  }

  if (pipeline_kv_update) {
    m_nPast += update_size;
  }

  *exec_time_total = exec_time;
  *wait_time_total = wait_time;
  return true;
}

bool QnnCpuModel::runInference(const std::vector<int32_t>& tokens, bool logits_all) {
  __DEBUG("qnn-cpu: run-inference start : n_tokens {}", tokens.size());

  auto start = std::chrono::steady_clock::now();

  // Technical note: int32_t can hold upto 596 hours
  // Even int16_t should be sufficient here - it holds upto 32.8 seconds
  int32_t total_wait_time = 0;
  int32_t total_exec_time = 0;

  // Setup inputs for inference
  setupInputTensors(tokens, false);

  auto& exec_models = model_order;
  if (!runInferenceHelper(exec_models, &total_wait_time, &total_exec_time, false, tokens.size()))
    return false;

  prev_run.num_tokens_processed = tokens.size();
  m_inference_count++;

  prev_run.was_bert_mode  = false;
  prev_run.was_logits_all = logits_all;

  auto stop = std::chrono::steady_clock::now();
  // QnnUtils::logProfile("Run Inference (cpp) took", start, stop);
  timeLogs["Run Inference (cpp) "].first += static_cast<double>(
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count());
  timeLogs["Run Inference (cpp) "].second++;
  QNN_DEBUG("[TIME] Wait[%d] Exec[%d]\n", total_wait_time, total_exec_time);
  return true;
}

bool QnnCpuModel::runInference(std::vector<uint8_t>& embeddings, bool logits_all) {
  __DEBUG("qnn-cpu: run-inference start : n_embd {}", embeddings.size());

  auto start = std::chrono::steady_clock::now();

  // Technical note: int32_t can hold upto 596 hours
  // Even int16_t should be sufficient here - it holds upto 32.8 seconds
  int32_t total_wait_time = 0;
  int32_t total_exec_time = 0;

  // Setup inputs for inference
  setupInputTensors(embeddings, false);

  const size_t embedBufSize = m_embeddingBufferSize;
  int32_t num_input_tokens  = embeddings.size() / embedBufSize;

  auto& exec_models = model_order;
  if (!runInferenceHelper(exec_models, &total_wait_time, &total_exec_time, false, num_input_tokens))
    return false;

  prev_run.num_tokens_processed = num_input_tokens;
  m_inference_count++;

  prev_run.was_bert_mode  = false;
  prev_run.was_logits_all = logits_all;

  auto stop = std::chrono::steady_clock::now();
  // QnnUtils::logProfile("Run Inference (cpp) took", start, stop);
  timeLogs["Run Inference (cpp) "].first += static_cast<double>(
      std::chrono::duration_cast<std::chrono::microseconds>(stop - start).count());
  timeLogs["Run Inference (cpp) "].second++;
  QNN_DEBUG("[TIME] Wait[%d] Exec[%d]\n", total_wait_time, total_exec_time);
  return true;
}

void QnnCpuModel::printFinalLogs() {
#if NSP_LOG_LEVEL > 1
  QNN_DEBUG("Total inference count : %d", m_inference_count);
  for (auto& [key, value] : timeLogs) {
    QNN_DEBUG("%s : %lf", key.c_str(), value.first / value.second);
  }
#endif
}

bool QnnCpuModel::setKVCacheNPast(size_t n_past) {
  if (n_past > m_nPast) {
    size_t num_update = n_past - m_nPast;
    if (n_past != 0 && num_update > prev_run.num_tokens_processed) {
      std::string err_msg = "Requested larger n_past update than #tokens produced by model";
      err_msg += std::to_string(num_update) + " vs. " + std::to_string(m_num_tokens);
      throw std::runtime_error(err_msg);
    }
  }

  m_nPast = n_past;
  return true;
}

size_t QnnCpuModel::getDequantLogits(std::vector<float>& dequant_logits, bool logits_all) {
  // if model is BERT, always return ALL logits
  if (model_output == ModelOutput::EMBEDDINGS) logits_all = true;

  __DEBUG("qnn-cpu: get-dequant-logits logits_all {}", logits_all);

  auto& logit_spec = m_output_specs[model_order.back()]["output_genAI"];
  float* logitBuf  = (float*)getBuffer(logit_spec);
  size_t offset    = 0;
  dequant_logits.clear();
  if (model_output == ModelOutput::LOGITS) {
    // if logits_all return [m_numLogits * m_vocab_size] else return [1 * m_vocab_size]
    if (!logits_all) {
      // Return the last processed token logits i.e. [ ..., [1]]
      if (m_numLogits > 1) {
        offset = (m_numLogits - 1) * m_vocab_size;
      }
    } else {
      // if m_numLogits > n_tokens_processed, it is left padded, [0, 0, [n_tokens_processed]]
      // calculate offset for getting the appropriate logits
      if (m_numLogits >= prev_run.num_tokens_processed) {
        offset = (m_numLogits - prev_run.num_tokens_processed) * m_vocab_size;
      }
    }
  }
#ifdef DUMP_LOGITS
  {
    char fname[255];
    sprintf(fname, "%s/logits/%03d", DEBUG_DUMP_TARGET_PATH, m_inference_count);
    QnnUtils::writeRawData(getBuffer(logit_spec), getBufferSize(logit_spec), fname);
  }
#endif
  if (model_output == ModelOutput::LOGITS) {
    // logits size = [m_numLogits * m_vocab_size]
    // logits might be left padded so, use calculated offset
    dequant_logits.reserve((getBufferSize(logit_spec) - (offset * sizeof(float))));
    for (auto i = offset; i < (getBufferSize(logit_spec) / sizeof(float)); ++i) {
      dequant_logits.push_back(logitBuf[i]);
    }
  } else if (model_output == ModelOutput::EMBEDDINGS) {
    // embeddings size = [n_tokens_processed * m_embd]
    dequant_logits.reserve((prev_run.num_tokens_processed * m_embd * sizeof(float)));
    for (auto i = offset; i < ((prev_run.num_tokens_processed * m_embd)); ++i) {
      dequant_logits.push_back(logitBuf[i]);
    }
  }

  return logits_all ? prev_run.num_tokens_processed : 1;
}

size_t QnnCpuModel::getLogits(Tensor& dequant_logits, bool logits_all) {
  // if model is BERT, always return ALL logits
  __DEBUG("qnn-cpu: get-dequant-logits logits_all {}", logits_all);

  auto& logit_spec = m_output_specs[model_order.back()]["output_genAI"];
  float* logitBuf  = (float*)getBuffer(logit_spec);
  size_t offset    = 0;

  // if logits_all return [m_numLogits * m_vocab_size] else return [1 * m_vocab_size]
  if (!logits_all) {
    // Return the last processed token logits i.e. [ ..., [1]]
    if (m_numLogits > 1) {
      offset = (m_numLogits - 1) * m_vocab_size;
    }
  } else {
    // if m_numLogits > n_tokens_processed, it is left padded, [0, 0, [n_tokens_processed]]
    // calculate offset for getting the appropriate logits
    if (m_numLogits >= prev_run.num_tokens_processed) {
      offset = (m_numLogits - prev_run.num_tokens_processed) * m_vocab_size;
    }
  }
#ifdef DUMP_LOGITS
  {
    char fname[255];
    sprintf(fname, "%s/logits/%03d", DEBUG_DUMP_TARGET_PATH, m_inference_count);
    QnnUtils::writeRawData(getBuffer(logit_spec), getBufferSize(logit_spec), fname);
  }
#endif
  // logits size = [m_numLogits * m_vocab_size]
  // logits might be left padded so, use calculated offset
  dequant_logits.setQuantizationParams(1, 0);
  dequant_logits.setSize((getBufferSize(logit_spec) / sizeof(float)) - offset);
  dequant_logits.setData((void*)(logitBuf + offset));
  dequant_logits.setDataType(TENSOR_DATATYPE_FLOAT_32);

  return logits_all ? prev_run.num_tokens_processed : 1;
}

bool QnnCpuModel::applyBinarySections(std::vector<std::string>& binsection_list) {
  // apply binary section for lora config
  for (int i = 0; i < binsection_list.size(); i++) {
    __DEBUG("qnn-cpu: applyBinarySections adapters {}", binsection_list.at(i));
    if (!m_qnnApi->applyBinarySection(i, binsection_list.at(i))) {
      __ERROR("qnn-cpu: Error in applyBinarySections {}", i);
      return false;
    }
  }
  return true;
}

bool QnnCpuModel::applyLoraStrength(const std::string& alpha_tensor_name, const float alpha_val) {
  for (auto it : m_loraConfig) {
    auto itt = std::find(it.second.alphas.begin(), it.second.alphas.end(), alpha_tensor_name);
    if (itt != it.second.alphas.end()) {
      m_lora_alpha_val[alpha_tensor_name] = alpha_val;
      return true;
    }
  }
  __ERROR("qnn-cpu: Could not find lora alpha tensor to apply");
  return false;
}

bool QnnCpuModel::applyLoraAdapter(const std::string& lora_adapter_name) {
  if (m_loraConfigType != LoraConfigType::LORA_ADAPTER_WEIGHT_ENABLE) {
    __ERROR("qnn-cpu: Lora config is not enable for adapters");
    return false;
  }

  if (!m_loraConfig.contains(lora_adapter_name)) {
    __ERROR("qnn-cpu: Could not find lora adapters config to apply ");
    return false;
  }

  m_adapter = lora_adapter_name;
  for (auto idx = 0; idx < m_loraConfig[lora_adapter_name].alpha_tensor_val.size(); idx++) {
    if (!applyLoraStrength(m_loraConfig[lora_adapter_name].alphas[idx],
                           m_lora_alpha_val[m_loraConfig[lora_adapter_name].alphas[idx]])) {
      __ERROR("qnn-cpu: Could not apply Alpha tensor ");
      return false;
    }
  }

  if (!applyBinarySections(m_loraConfig[lora_adapter_name].binsection_list)) {
    __ERROR("qnn-cpu: Could not apply binary Sections ");
    return false;
  }
  return true;
}

// TODO: implement save/restore
size_t QnnCpuModel::loadKVCache(const std::string& load_path) {
  // TO read the cache file into KV tensor
  std::ifstream f(load_path, std::ios::in | std::ios::binary);
  if (f.fail()) {
    // TODO: replace with proper error handling
    __ERROR("qnn-cpu: load-kv errror reading file {}", load_path);
    return 0;
  }

  CacheFileSpec spec;
  f.read((char*)&spec, sizeof(spec));
  if (spec.magic != 0xC0DE) {
    __ERROR("qnn-cpu: load-kv expected 0xC0DE found {:#x}", spec.magic);
    return 0;
  }
  __DEBUG(
      "qnn-cpu: load-kv {{ num_tensors {}, magic {}, dtype {}, n_heads {}, embed_dim {} "
      "update_size {} }}",
      spec.num_tensors,
      spec.magic,
      int(spec.dtype),
      spec.n_heads,
      spec.embed_dim,
      spec.update_size);

  const int32_t n_valid        = static_cast<int32_t>(spec.update_size);
  const size_t copy_size       = n_valid * m_head_dim;
  const size_t skip_size       = (m_ctx_size + 1) * m_head_dim;
  const size_t copy_block_size = n_valid * (m_head_dim / 32);
  const size_t skip_block_size = (m_ctx_size + 1) * (m_head_dim / 32);

  if (!m_kv_quant) {
    float* input_id_k_cache_buffer = (float*)getBuffer(t_input_ids_k_cache);
    float* input_id_v_cache_buffer = (float*)getBuffer(t_input_ids_v_cache);
    // K$, V$ 4D Tensor {n_layer, n_kv_heads, n_ctx, n_head_dim}

    for (int i = 0; i < m_num_layer; i++) {
      for (int j = 0; j < m_num_kv_heads; j++) {
        f.read((char*)input_id_k_cache_buffer, copy_size * sizeof(float));
        input_id_k_cache_buffer += skip_size;
      }
    }

    for (int i = 0; i < m_num_layer; i++) {
      for (int j = 0; j < m_num_kv_heads; j++) {
        f.read((char*)input_id_v_cache_buffer, copy_size * sizeof(float));
        input_id_v_cache_buffer += skip_size;
      }
    }
  } else {
    int8_t* input_id_k_cache_buffer = (int8_t*)getBuffer(t_input_ids_k_cache);
    int8_t* input_id_v_cache_buffer = (int8_t*)getBuffer(t_input_ids_v_cache);

    float* input_id_k_cache_scale_buffer = (float*)getBuffer(t_input_ids_k_scale);
    float* input_id_v_cache_scale_buffer = (float*)getBuffer(t_input_ids_v_scale);

    // read KV$
    for (int i = 0; i < m_num_layer; i++) {
      for (int j = 0; j < m_num_kv_heads; j++) {
        f.read((char*)input_id_k_cache_buffer, copy_size * sizeof(int8_t));
        input_id_k_cache_buffer += skip_size;
      }
    }

    for (int i = 0; i < m_num_layer; i++) {
      for (int j = 0; j < m_num_kv_heads; j++) {
        f.read((char*)input_id_v_cache_buffer, copy_size * sizeof(int8_t));
        input_id_v_cache_buffer += skip_size;
      }
    }

    // read scales
    for (int i = 0; i < m_num_layer; i++) {
      for (int j = 0; j < m_num_kv_heads; j++) {
        f.read((char*)input_id_k_cache_scale_buffer, copy_block_size * sizeof(float));
        input_id_k_cache_scale_buffer += skip_block_size;
      }
    }
    for (int i = 0; i < m_num_layer; i++) {
      for (int j = 0; j < m_num_kv_heads; j++) {
        f.read((char*)input_id_v_cache_scale_buffer, copy_block_size * sizeof(float));
        input_id_v_cache_scale_buffer += skip_block_size;
      }
    }
  }

  f.close();

  m_nPast                       = n_valid;
  prev_run.num_tokens_processed = m_nPast;
  return spec.update_size;
}

bool QnnCpuModel::saveKVCache(const std::string& save_path) {
  __DEBUG("qnn-cpu: save-kv path {}", save_path);

  std::ofstream f(save_path, std::ios::out | std::ios::binary);
  if (f.fail()) {
    __ERROR("qnn-cpu: save-kv error opening file : {}", save_path);
    throw std::runtime_error("Failed to write to cache file. Please re-check path");
  }

  const uint32_t n_valid              = static_cast<uint32_t>(m_nPast);
  const CacheFileSpec::DataType dtype = CacheFileSpec::DataType::FLOAT32_T;

  // Save the cache file metadata
  CacheFileSpec spec(m_num_layer * 2, 0xc0de, dtype, 0x0, m_num_heads, m_head_dim, n_valid);
  f.write((char*)&spec, sizeof(spec));  // as nsp already updated the spec

  const size_t copy_size       = n_valid * m_head_dim;
  const size_t skip_size       = (m_ctx_size + 1) * m_head_dim;
  const size_t copy_block_size = n_valid * (m_head_dim / 32);
  const size_t skip_block_size = (m_ctx_size + 1) * (m_head_dim / 32);

  if (n_valid > 0) {
    // Dump KeyCache and ValueCache
    if (!m_kv_quant) {
      float* input_id_k_cache_buffer = (float*)getBuffer(t_input_ids_k_cache);
      float* input_id_v_cache_buffer = (float*)getBuffer(t_input_ids_v_cache);

      // K$, V$ 4D Tensor {n_layer, n_kv_heads, n_ctx, n_head_dim}

      for (int i = 0; i < m_num_layer; i++) {
        for (int j = 0; j < m_num_kv_heads; j++) {
          f.write((char*)input_id_k_cache_buffer, copy_size * sizeof(float));
          input_id_k_cache_buffer += skip_size;
        }
      }

      for (int i = 0; i < m_num_layer; i++) {
        for (int j = 0; j < m_num_kv_heads; j++) {
          f.write((char*)input_id_v_cache_buffer, copy_size * sizeof(float));
          input_id_v_cache_buffer += skip_size;
        }
      }
    } else {
      int8_t* input_id_k_cache_buffer = (int8_t*)getBuffer(t_input_ids_k_cache);
      int8_t* input_id_v_cache_buffer = (int8_t*)getBuffer(t_input_ids_v_cache);

      float* input_id_k_cache_scale_buffer = (float*)getBuffer(t_input_ids_k_scale);
      float* input_id_v_cache_scale_buffer = (float*)getBuffer(t_input_ids_v_scale);

      for (int i = 0; i < m_num_layer; i++) {
        for (int j = 0; j < m_num_kv_heads; j++) {
          f.write((char*)input_id_k_cache_buffer, copy_size * sizeof(int8_t));
          input_id_k_cache_buffer += skip_size;
        }
      }

      for (int i = 0; i < m_num_layer; i++) {
        for (int j = 0; j < m_num_kv_heads; j++) {
          f.write((char*)input_id_v_cache_buffer, copy_size * sizeof(int8_t));
          input_id_v_cache_buffer += skip_size;
        }
      }

      // write scales
      for (int i = 0; i < m_num_layer; i++) {
        for (int j = 0; j < m_num_kv_heads; j++) {
          f.write((char*)input_id_k_cache_scale_buffer, copy_block_size * sizeof(float));
          input_id_k_cache_scale_buffer += skip_block_size;
        }
      }
      for (int i = 0; i < m_num_layer; i++) {
        for (int j = 0; j < m_num_kv_heads; j++) {
          f.write((char*)input_id_v_cache_scale_buffer, copy_block_size * sizeof(float));
          input_id_v_cache_scale_buffer += skip_block_size;
        }
      }
    }
  }
  f.flush();
  f.close();

  return true;
}

#if __ARM_NEON__ || __ARM_NEON || (_MSC_VER && (_M_ARM || _M_ARM64 || _M_ARM64EC))
#include <arm_neon.h>

bool QnnCpuModel::setKVQuantHead(
    CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) {
  uint32_t context_size = m_ctx_size;
  uint32_t n_head       = spec.n_heads;
  uint32_t kv_dim       = spec.embed_dim;
  uint32_t n_tok        = spec.update_size;

  std::vector<float> kv_data(2 * n_tok * kv_dim);
  float* k_reference = (float*)kv_data.data();
  float* v_reference = (float*)kv_data.data() + (n_tok * kv_dim);

  uint8_t* k_buffer = (uint8_t*)data;
  for (uint32_t l = 0; l < n_tok; l++) {
    int k_len  = kv_dim / 2;
    uint32_t k = 0;

    uint8x8_t zero_point  = vdup_n_u8(128);
    float32x4_t scale_vec = vdupq_n_f32(((float)scale[0]));

    // Interleave K$
    // QNN HTP: [0 2 4 ... 126 1 3 5 ... 127]
    // QNN CPU: [0 1 2 ... 63  64 65 ... 127]
    for (; k <= (k_len - 8); k += 8) {
      uint32_t write_loc = l * kv_dim + 2 * k;

      uint8x8_t k_low           = vld1_u8(&k_buffer[l * kv_dim + k]);
      uint8x8_t k_high          = vld1_u8(&k_buffer[l * kv_dim + k_len + k]);
      uint8x8x2_t interleaved_k = vzip_u8(k_low, k_high);

      for (uint32_t m = 0; m < 2; m++) {
        int16x8_t k_i16 = vmovl_s8(vreinterpret_s8_u8(vadd_u8(interleaved_k.val[m], zero_point)));
        float32x4_t k_low_f32  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(k_i16)));
        float32x4_t k_high_f32 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(k_i16)));
        float32x4_t dq_k_low   = vmulq_f32(k_low_f32, scale_vec);
        float32x4_t dq_k_high  = vmulq_f32(k_high_f32, scale_vec);
        vst1q_f32(&k_reference[write_loc], dq_k_low);
        vst1q_f32(&k_reference[write_loc + 4], dq_k_high);
        write_loc += 8;
      }
    }

    // Handle remaining elements if any
    for (; k < k_len; k++) {
      const uint32_t read_loc  = l * kv_dim + k;
      const uint32_t write_loc = l * kv_dim + 2 * k;
      k_reference[write_loc]   = (static_cast<float>(k_buffer[read_loc]) - 128) * ((float)scale[0]);
      k_reference[write_loc + 1] =
          (static_cast<float>(k_buffer[read_loc + k_len]) - 128) * ((float)scale[0]);
    }
  }

  uint8_t* v_buffer = ((uint8_t*)data) + (n_tok * kv_dim);
  for (uint32_t l = 0; l < n_tok; l++) {
    uint32_t offset = l * kv_dim;
    uint32_t k      = 0;

    uint8x16_t zero_point       = vdupq_n_u8(128);
    const float32x4_t scale_vec = vdupq_n_f32((float)scale[1]);

    for (; k + 15 < kv_dim; k += 16) {
      int8x16_t input_s8 =
          vreinterpretq_s8_u8(vaddq_u8(vld1q_u8(&v_buffer[offset + k]), zero_point));
      int16x8_t input_s16_low  = vmovl_s8(vget_low_s8(input_s8));
      int16x8_t input_s16_high = vmovl_s8(vget_high_s8(input_s8));

      float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(input_s16_low))), scale_vec);
      float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(input_s16_low))), scale_vec);
      float32x4_t f2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(input_s16_high))), scale_vec);
      float32x4_t f3 =
          vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(input_s16_high))), scale_vec);

      vst1q_f32(&v_reference[offset + k], f0);
      vst1q_f32(&v_reference[offset + k + 4], f1);
      vst1q_f32(&v_reference[offset + k + 8], f2);
      vst1q_f32(&v_reference[offset + k + 12], f3);
    }

    // Handle remaining elements if any
    for (; k < kv_dim; k++) {
      uint32_t loc     = offset + k;
      v_reference[loc] = (static_cast<float>(v_buffer[loc]) - 128) * scale[1];
    }
  }

  const uint32_t block_size = 32;
  const uint32_t ivec_size  = 16;
  const uint32_t fvec_size  = 4;
  uint32_t layer_size       = n_head * (context_size + 1) * kv_dim;
  uint32_t head_size        = (context_size + 1) * kv_dim;
  uint32_t global_loc       = layer * layer_size + head * head_size;
  uint32_t global_scale_loc = global_loc / block_size;
  int8_t* k_quant           = (int8_t*)getBuffer(t_input_ids_k_cache);
  int8_t* v_quant           = (int8_t*)getBuffer(t_input_ids_v_cache);
  float* k_scale            = (float*)getBuffer(t_input_ids_k_scale);
  float* v_scale            = (float*)getBuffer(t_input_ids_v_scale);

  for (uint32_t l = 0; l < n_tok; l++) {
    for (uint32_t k = 0; k < kv_dim / block_size; k++) {
      const uint32_t quant_loc = l * kv_dim;
      const uint32_t scale_loc = l * (kv_dim / block_size) + k;

      float32x4_t k_val[8], v_val[8];
      float32x4_t k_abs[8], v_abs[8];

      for (int m = 0; m < 8; m++) {
        k_val[m] = vld1q_f32(&k_reference[quant_loc + k * block_size + m * fvec_size]);
        k_abs[m] = vabsq_f32(k_val[m]);
        v_val[m] = vld1q_f32(&v_reference[quant_loc + k * block_size + m * fvec_size]);
        v_abs[m] = vabsq_f32(v_val[m]);
      }

      k_abs[0]   = vmaxq_f32(k_abs[0], k_abs[1]);
      k_abs[2]   = vmaxq_f32(k_abs[2], k_abs[3]);
      k_abs[4]   = vmaxq_f32(k_abs[4], k_abs[5]);
      k_abs[6]   = vmaxq_f32(k_abs[6], k_abs[7]);
      k_abs[0]   = vmaxq_f32(k_abs[0], k_abs[2]);
      k_abs[4]   = vmaxq_f32(k_abs[4], k_abs[6]);
      k_abs[0]   = vmaxq_f32(k_abs[0], k_abs[4]);
      float kmax = vmaxvq_f32(k_abs[0]);

      v_abs[0]   = vmaxq_f32(v_abs[0], v_abs[1]);
      v_abs[2]   = vmaxq_f32(v_abs[2], v_abs[3]);
      v_abs[4]   = vmaxq_f32(v_abs[4], v_abs[5]);
      v_abs[6]   = vmaxq_f32(v_abs[6], v_abs[7]);
      v_abs[0]   = vmaxq_f32(v_abs[0], v_abs[2]);
      v_abs[4]   = vmaxq_f32(v_abs[4], v_abs[6]);
      v_abs[0]   = vmaxq_f32(v_abs[0], v_abs[4]);
      float vmax = vmaxvq_f32(v_abs[0]);

      const float dk                        = kmax / ((1 << 7) - 1);
      const float idk                       = dk ? 1.f / dk : 0.f;
      k_scale[global_scale_loc + scale_loc] = dk;

      const float dv                        = vmax / ((1 << 7) - 1);
      const float idv                       = dv ? 1.f / dv : 0.f;
      v_scale[global_scale_loc + scale_loc] = dv;

      for (int m = 0; m < 2; m++) {
        float32x4_t k0  = vmulq_n_f32(k_val[(m * 4) + 0], idk);
        float32x4_t k1  = vmulq_n_f32(k_val[(m * 4) + 1], idk);
        float32x4_t k2  = vmulq_n_f32(k_val[(m * 4) + 2], idk);
        float32x4_t k3  = vmulq_n_f32(k_val[(m * 4) + 3], idk);
        int8x8_t k01_i8 = vqmovn_s16(
            vcombine_s16(vqmovn_s32(vcvtaq_s32_f32(k0)), vqmovn_s32(vcvtaq_s32_f32(k1))));
        int8x8_t k23_i8 = vqmovn_s16(
            vcombine_s16(vqmovn_s32(vcvtaq_s32_f32(k2)), vqmovn_s32(vcvtaq_s32_f32(k3))));
        int8x16_t k_q = vcombine_s8(k01_i8, k23_i8);
        vst1q_s8(&k_quant[global_loc + quant_loc + k * block_size + m * ivec_size], k_q);

        float32x4_t v0  = vmulq_n_f32(v_val[(m * 4) + 0], idv);
        float32x4_t v1  = vmulq_n_f32(v_val[(m * 4) + 1], idv);
        float32x4_t v2  = vmulq_n_f32(v_val[(m * 4) + 2], idv);
        float32x4_t v3  = vmulq_n_f32(v_val[(m * 4) + 3], idv);
        int8x8_t v01_i8 = vqmovn_s16(
            vcombine_s16(vqmovn_s32(vcvtaq_s32_f32(v0)), vqmovn_s32(vcvtaq_s32_f32(v1))));
        int8x8_t v23_i8 = vqmovn_s16(
            vcombine_s16(vqmovn_s32(vcvtaq_s32_f32(v2)), vqmovn_s32(vcvtaq_s32_f32(v3))));
        int8x16_t v_q = vcombine_s8(v01_i8, v23_i8);
        vst1q_s8(&v_quant[global_loc + quant_loc + k * block_size + m * ivec_size], v_q);
      }
    }
  }

  m_nPast                       = n_tok;
  prev_run.num_tokens_processed = m_nPast;
  return true;
}

bool QnnCpuModel::setKVHead(
    CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) {
  if (m_kv_quant) return setKVQuantHead(spec, layer, head, data, scale);

  float* k_reference    = (float*)getBuffer(t_input_ids_k_cache);
  float* v_reference    = (float*)getBuffer(t_input_ids_v_cache);
  uint32_t context_size = m_ctx_size;
  uint32_t n_head       = spec.n_heads;
  uint32_t kv_dim       = spec.embed_dim;
  uint32_t n_tok        = spec.update_size;
  uint32_t layer_size   = n_head * (context_size + 1) * kv_dim;
  uint32_t head_size    = (context_size + 1) * kv_dim;
  uint32_t global_loc   = layer * layer_size + head * head_size;

  uint8_t* k_buffer = (uint8_t*)data;
  for (uint32_t l = 0; l < n_tok; l++) {
    int k_len  = kv_dim / 2;
    uint32_t k = 0;

    uint8x8_t zero_point  = vdup_n_u8(128);
    float32x4_t scale_vec = vdupq_n_f32(((float)scale[0]));

    // Interleave K$
    // QNN HTP: [0 2 4 ... 126 1 3 5 ... 127]
    // QNN CPU: [0 1 2 ... 63  64 65 ... 127]
    for (; k <= (k_len - 8); k += 8) {
      uint32_t write_loc = l * kv_dim + 2 * k;

      uint8x8_t k_low           = vld1_u8(&k_buffer[l * kv_dim + k]);
      uint8x8_t k_high          = vld1_u8(&k_buffer[l * kv_dim + k_len + k]);
      uint8x8x2_t interleaved_k = vzip_u8(k_low, k_high);

      for (uint32_t m = 0; m < 2; m++) {
        int16x8_t k_i16 = vmovl_s8(vreinterpret_s8_u8(vadd_u8(interleaved_k.val[m], zero_point)));
        float32x4_t k_low_f32  = vcvtq_f32_s32(vmovl_s16(vget_low_s16(k_i16)));
        float32x4_t k_high_f32 = vcvtq_f32_s32(vmovl_s16(vget_high_s16(k_i16)));
        float32x4_t dq_k_low   = vmulq_f32(k_low_f32, scale_vec);
        float32x4_t dq_k_high  = vmulq_f32(k_high_f32, scale_vec);
        vst1q_f32(&k_reference[global_loc + write_loc], dq_k_low);
        vst1q_f32(&k_reference[global_loc + write_loc + 4], dq_k_high);
        write_loc += 8;
      }
    }

    // Handle remaining elements if any
    for (; k < k_len; k++) {
      const uint32_t read_loc  = l * kv_dim + k;
      const uint32_t write_loc = l * kv_dim + 2 * k;
      k_reference[global_loc + write_loc] =
          (static_cast<float>(k_buffer[read_loc]) - 128) * ((float)scale[0]);
      k_reference[global_loc + write_loc + 1] =
          (static_cast<float>(k_buffer[read_loc + k_len]) - 128) * ((float)scale[0]);
    }
  }

  uint8_t* v_buffer = ((uint8_t*)data) + (n_tok * kv_dim);
  for (uint32_t l = 0; l < n_tok; l++) {
    uint32_t offset = l * kv_dim;
    uint32_t k      = 0;

    uint8x16_t zero_point       = vdupq_n_u8(128);
    const float32x4_t scale_vec = vdupq_n_f32((float)scale[1]);

    for (; k + 15 < kv_dim; k += 16) {
      int8x16_t input_s8 =
          vreinterpretq_s8_u8(vaddq_u8(vld1q_u8(&v_buffer[offset + k]), zero_point));
      int16x8_t input_s16_low  = vmovl_s8(vget_low_s8(input_s8));
      int16x8_t input_s16_high = vmovl_s8(vget_high_s8(input_s8));

      float32x4_t f0 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(input_s16_low))), scale_vec);
      float32x4_t f1 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(input_s16_low))), scale_vec);
      float32x4_t f2 = vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(input_s16_high))), scale_vec);
      float32x4_t f3 =
          vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_high_s16(input_s16_high))), scale_vec);

      vst1q_f32(&v_reference[global_loc + offset + k], f0);
      vst1q_f32(&v_reference[global_loc + offset + k + 4], f1);
      vst1q_f32(&v_reference[global_loc + offset + k + 8], f2);
      vst1q_f32(&v_reference[global_loc + offset + k + 12], f3);
    }

    // Handle remaining elements if any
    for (; k < kv_dim; k++) {
      uint32_t loc                  = offset + k;
      v_reference[global_loc + loc] = (static_cast<float>(v_buffer[loc]) - 128) * scale[1];
    }
  }

  m_nPast                       = n_tok;
  prev_run.num_tokens_processed = m_nPast;
  return true;
}
#else
bool QnnCpuModel::setKVQuantHead(
    CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) {
  uint32_t context_size = m_ctx_size;
  uint32_t n_head = spec.n_heads;
  uint32_t kv_dim = spec.embed_dim;
  uint32_t n_tok = spec.update_size;

  std::vector<float> kv_data(2 * n_tok * kv_dim);
  float* k_reference = (float*)kv_data.data();
  float* v_reference = (float*)kv_data.data() + (n_tok * kv_dim);

  uint8_t* k_buffer = (uint8_t*)data;
  for (uint32_t l = 0; l < n_tok; l++) {
    for (uint32_t k = 0; k < kv_dim; k++) {
      // Interleave K$
      // QNN HTP: [0 2 4 ... 126 1 3 5 ... 127]
      // QNN CPU: [0 1 2 ... 63  64 65 ... 127]
      const uint32_t interleaved_k = (2 * k < kv_dim) ? 2 * k : 2 * (k - kv_dim / 2) + 1;
      //  For ScopGPT KV$ Format
      const uint32_t read_loc = l * kv_dim + k;
      const uint32_t write_loc = l * kv_dim + interleaved_k;
      k_reference[write_loc] = (static_cast<float>(k_buffer[read_loc]) - 128) * scale[0];
    }
  }

  uint8_t* v_buffer = ((uint8_t*)data) + (n_tok * kv_dim);
  for (uint32_t l = 0; l < n_tok; l++) {
    for (uint32_t k = 0; k < kv_dim; k++) {
      const uint32_t read_loc = l * kv_dim + k;
      const uint32_t write_loc = l * kv_dim + k;
      v_reference[write_loc] = (static_cast<float>(v_buffer[read_loc]) - 128) * scale[1];
    }
  }

  const uint32_t block_size = 32;
  uint32_t layer_size = n_head * (context_size + 1) * kv_dim;
  uint32_t head_size = (context_size + 1) * kv_dim;
  uint32_t global_loc = layer * layer_size + head * head_size;
  uint32_t global_scale_loc = global_loc / block_size;
  int8_t* k_quant = (int8_t*)getBuffer(t_input_ids_k_cache);
  int8_t* v_quant = (int8_t*)getBuffer(t_input_ids_v_cache);
  float* k_scale = (float*)getBuffer(t_input_ids_k_scale);
  float* v_scale = (float*)getBuffer(t_input_ids_v_scale);

  for (uint32_t l = 0; l < n_tok; l++) {
#pragma clang loop vectorize(enable)
    for (uint32_t k = 0; k < kv_dim / block_size; k++) {
      const uint32_t quant_loc = l * kv_dim;
      const uint32_t scale_loc = l * (kv_dim / block_size) + k;

      float kmax = 0.f;
      float vmax = 0.f;

      for (size_t m = 0; m < block_size; m++) {
        float kval = fabs((float)k_reference[quant_loc + k * block_size + m]);
        kmax = fmax(kmax, kval);

        float vval = fabs((float)v_reference[quant_loc + k * block_size + m]);
        vmax = fmax(vmax, vval);
      }

      const float dk = kmax / ((1 << 7) - 1);
      const float idk = dk ? 1.f / dk : 0.f;
      k_scale[global_scale_loc + scale_loc] = dk;

      const float dv = vmax / ((1 << 7) - 1);
      const float idv = dv ? 1.f / dv : 0.f;
      v_scale[global_scale_loc + scale_loc] = dv;

      for (size_t m = 0; m < block_size; m++) {
        k_quant[global_loc + quant_loc + k * block_size + m] =
            (int8_t)roundf(k_reference[quant_loc + k * block_size + m] * idk);

        v_quant[global_loc + quant_loc + k * block_size + m] =
            (int8_t)roundf(v_reference[quant_loc + k * block_size + m] * idv);
      }
    }
  }

  m_nPast = n_tok;
  prev_run.num_tokens_processed = m_nPast;
  return true;
}

bool QnnCpuModel::setKVHead(
    CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) {
  if (m_kv_quant) return setKVQuantHead(spec, layer, head, data, scale);

  float* k_reference = (float*)getBuffer(t_input_ids_k_cache);
  float* v_reference = (float*)getBuffer(t_input_ids_v_cache);
  uint32_t context_size = m_ctx_size;
  uint32_t n_head = spec.n_heads;
  uint32_t kv_dim = spec.embed_dim;
  uint32_t n_tok = spec.update_size;
  uint32_t layer_size = n_head * (context_size + 1) * kv_dim;
  uint32_t head_size = (context_size + 1) * kv_dim;
  uint32_t global_loc = layer * layer_size + head * head_size;

  uint8_t* k_buffer = (uint8_t*)data;
  for (uint32_t l = 0; l < n_tok; l++) {
    for (uint32_t k = 0; k < kv_dim; k++) {
      // Interleave K$
      // QNN HTP: [0 2 4 ... 126 1 3 5 ... 127]
      // QNN CPU: [0 1 2 ... 63  64 65 ... 127]
      const uint32_t interleaved_k = (2 * k < kv_dim) ? 2 * k : 2 * (k - kv_dim / 2) + 1;
      //  For ScopGPT KV$ Format
      const uint32_t read_loc = l * kv_dim + k;
      const uint32_t write_loc = l * kv_dim + interleaved_k;
      k_reference[write_loc + global_loc] =
          (static_cast<float>(k_buffer[read_loc]) - 128) * scale[0];
    }
  }

  uint8_t* v_buffer = ((uint8_t*)data) + (n_tok * kv_dim);
  for (uint32_t l = 0; l < n_tok; l++) {
    for (uint32_t k = 0; k < kv_dim; k++) {
      const uint32_t read_loc = l * kv_dim + k;
      const uint32_t write_loc = l * kv_dim + k;
      v_reference[write_loc + global_loc] =
          (static_cast<float>(v_buffer[read_loc]) - 128) * scale[1];
    }
  }

  m_nPast = n_tok;
  prev_run.num_tokens_processed = m_nPast;
  return true;
}
#endif

}  // namespace qualla
