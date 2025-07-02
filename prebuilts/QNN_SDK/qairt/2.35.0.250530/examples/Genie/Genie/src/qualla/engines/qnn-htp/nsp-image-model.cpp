//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#define _USE_MATH_DEFINES  // Used for M_PI

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <span>
#include <sstream>

#include "fmt/format.h"
#include "fmt/os.h"
#include "fmt/ranges.h"
#include "fp16/fp16.h"
#include "nsp-image-model.hpp"
#include "qualla/detail/cache-file.hpp"
#include "qualla/detail/timer.hpp"
#include "qualla/env.hpp"

namespace fs = std::filesystem;

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
namespace qualla {

QnnNspImageModel::QnnNspImageModel(Env& env, const QnnNspBaseModel::Params& params)
    : QnnNspBaseModel(env, params) {
  // Initialize QnnAPI
  m_qnnApi                = std::unique_ptr<QnnApi>(new QnnApi());
  spill_fill_buffer_size  = params.spill_fill_bufsize;
  m_use_mmap              = params.use_mmap;
  m_use_async_Init        = params.use_async_Init;
  mmap_budget             = params.mmap_budget;
  graph_switching         = params.graph_switching;
  load_select_graphs      = params.load_select_graphs;
  m_modelArchitectureType = params.modelArchitectureType;

  if (graph_switching && !m_use_mmap)
    __WARN("Graph switching with non-mmaped implementation can cause high sustained memory usage");

  exec_select_graphs = params.exec_select_graphs;
  if (!exec_select_graphs.empty())
    __DEBUG("qnn-htp : Execute selected graphs = {}", exec_select_graphs);

  // Set up filename list.
  for (auto& i : params.model_list) {
    fs::path model_path = fs::path(i);
    if (model_path.is_relative()) model_path = model_basedir / fs::path(i);
    if (!fs::is_regular_file(model_path)) {
      __ERROR("NSPModel: Can't access model file : {}", model_path.string());
      throw std::runtime_error("NSPModel: Can't access model file : " + model_path.string());
    }
    model_filelist.push_back(model_path.string());
  }

  if (lora_conf != LoraConfigType::LORA_DISABLE) {
    lora_config.insert(params.lora_param.begin(), params.lora_param.end());
  }

  if (params.n_threads > 0) {
    _threaded = true;
    _cpumask  = params.cpumask;
    __DEBUG("qnn-htp: starting threadpool : n_threads {} params. {:#x} poll {}",
            params.n_threads,
            _cpumask,
            params.poll);
    threadpool.start(params.n_threads, _cpumask, params.poll);
  }

  // Initialize QNN IO Tensor
  m_ioTensor = std::unique_ptr<IOTensor>(
      new IOTensor(m_sharedBuffer ? BufferAlloc::SHARED_BUFFER : BufferAlloc::DEFAULT,
                   m_sharedBuffer ? m_qnnApi->getQnnInterfaceVer() : nullptr));

  m_qnnApi->setIOTensorBufferMgr(m_ioTensor.get());

  if (params.debug_specs || params.debug_tensors) {
    if (!fs::exists(params.debug_path) && !fs::create_directories(params.debug_path))
      throw std::runtime_error("Could not create debug directory : " + params.debug_path);
  }
}

QnnNspImageModel::~QnnNspImageModel() {
  qualla::Timer start;

  if (_threaded) {
    __DEBUG("qnn-htp: stopping threadpool");
    threadpool.stop();  // Stop Threadpool first
  }

  __DEBUG("qnn-htp: model destruct complete: {} usec", start.elapsed_usec());
}

// Given a filename, initializeModel load and initializes QNN runtime libraries and the model
bool QnnNspImageModel::initializeModel(void) {
  qualla::Timer start;

  __DEBUG("qnn-htp: model init start");

  // Default backends
#ifdef _WIN32
  const std::string m_backend                = _backend_lib.empty() ? "QnnHtp.dll" : _backend_lib;
  const std::string m_systemLib              = "QnnSystem.dll";
  const std::string backendExtensionsLibPath = "QnnHtpNetRunExtensions.dll";
#else
  const std::string m_backend                = _backend_lib.empty() ? "libQnnHtp.so" : _backend_lib;
  const std::string m_systemLib              = "libQnnSystem.so";
  const std::string backendExtensionsLibPath = "libQnnHtpNetRunExtensions.so";
#endif
#ifdef QUALLA_INTERNAL_QNN_SDK
  if (_backend_ext_conf.empty()) {
    __INFO("No backend extension config provided");
  }
  fs::path m_backendExtensionsConfigPath = fs::path(_backend_ext_conf);
#else
  fs::path m_backendExtensionsConfigPath     = _backend_ext_conf.empty()
                                                   ? fs::path("data") / "htp_backend_ext_config.json"
                                                   : fs::path(_backend_ext_conf);

  if (m_backendExtensionsConfigPath.is_relative())
    m_backendExtensionsConfigPath = fs::path(model_basedir) / m_backendExtensionsConfigPath;

  if (!fs::is_regular_file(m_backendExtensionsConfigPath)) {
    __ERROR("Cannot access {}", m_backendExtensionsConfigPath.string());
    return false;
  }
#endif
  __INFO("Backend library : {}", m_backend);
  __INFO("System library  : {}", m_systemLib);
  __INFO("Model dir   : {}", model_basedir.string());
  __INFO("Model files : {}", model_filelist);
  __INFO("Backend extensions lib path : {}", backendExtensionsLibPath);
  __INFO("Backend extensions config path : {}", m_backendExtensionsConfigPath.string());

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
  if (!m_qnnApi->initializeHtp(m_backend,
                               model_filelist,
                               BackendExtensionsConfigs(backendExtensionsLibPath,
                                                        m_backendExtensionsConfigPath.string()),
                               qnn::tools::netrun::PerfProfile::BURST,
                               {},           // graphConfigs
                               true,         // loadFromCachedBinary
                               m_systemLib,  // systemLibraryPath
                               false,
                               spill_fill_buffer_size,
                               m_dataAlignmentSize,
                               m_use_mmap,
                               m_use_async_Init,
                               mmap_budget,
                               _debug_qnn,
                               graph_switching,
                               exec_select_graphs,
                               load_select_graphs,
                               logLevel,
                               logCallback)) {
    __ERROR("qnn-api initialization failed!");
    return false;
  }

  int32_t n_splits = 0;
  m_num_graphs     = m_qnnApi->getGraphsCount();
  n_splits         = m_num_graphs;
  __INFO("qnn-api initialized with {} graph(s)", m_num_graphs);
  auto graphs_info = m_qnnApi->getGraphsInfo();
  m_variant_list.reserve(m_num_graphs);
  // Create NSPGraph for each splits
  m_nsp_graphs.reserve(n_splits);
  std::map<int32_t, std::vector<std::string>> graph_names;
  for (size_t graph_idx = 0; graph_idx < m_num_graphs; graph_idx++) {
    qnn_wrapper_api::GraphInfo_t* const graph_info = graphs_info[graph_idx];
    GraphVariant graph(graph_info, m_qnnApi->getContexts(graph_info), m_layerNames, _env);
    graph.n_tokens = 0;
    __DEBUG("qnn-htp: Graph {}", graph.graph_name);

    if (exec_select_graphs.size() != 0 &&
        std::find(exec_select_graphs.begin(), exec_select_graphs.end(), graph.graph_name) ==
            exec_select_graphs.end()) {
      __DEBUG("qnn-htp: Graph {} is not selected to execute based on conf file", graph.graph_name);
      continue;
    }
    m_variant_list.emplace_back(graph);
    graph_names[n_splits].push_back(graph.graph_name);
    m_graph_map[std::string(graph_info->graphName)] = &m_variant_list.back();
    m_nsp_graphs.emplace_back(graph_idx, _env, m_qnnApi.get(), m_ioTensor.get());
    m_nsp_graphs.back().setDebugMode(_debug_specs, _debug_tensors, _debug_path);
  }

  if (exec_select_graphs.size() != 0 && graph_names.empty()) {
    __ERROR("No matching graphs based on conf file");
  }

  // Insert all GraphVariants into corresponding NSPGraph
  for (auto& [input_size, graphs] : graph_names) {
    std::sort(graphs.begin(), graphs.end());
    for (int idx = 0; idx < graphs.size(); idx++)
      m_nsp_graphs.at(idx).addGraph(m_graph_map.at(graphs[idx]));
  }

  if (_debug_specs) dumpTensorSpecs();

  __DEBUG("qnn-htp: Model Init complete: {} usec", start.elapsed_usec());

  return true;
}

// Once the model has been loaded, initialize IO Tensors
// m_ioTensors is initialized by the context for now
bool QnnNspImageModel::initializeIOTensors() {
  if (m_use_async_Init == false) {  // IO Tensor Mem Registration is already done within the
                                    // model_initailize by Qnn_API for Sync Init.

    // set loraWeights Enabled
    _lora_enabled = m_qnnApi->getLoraWeightEnabled();
    for (QnnNspGraph& graph : m_nsp_graphs) {
      // TensorAllocInfo is added to each NSP graph.
      // Needed by Pointer_SHIFT Registration During Execute.
      graph.tensor_alloc_info = m_qnnApi->getTensorAllocInfo();
      if (graph.tensor_alloc_info == NULL) {
        __ERROR("Error Tensor Allocation Failed.");
        return false;
      }
    }
    return true;
  }

  // This path is used in case of use Async Init is true.
  qualla::Timer start;

  __DEBUG("qnn-htp: init IO tensors start");

  // Ideally, we should create and initalize m_ioTensor for each context, but we want to
  // be able to see/use all the buffers in every contexts so that they can be connected
  // with each other. Hence, we are using only the first context to initialize the m_ioTensor
  // and use it for all graphs/contexts.
  __DEBUG("qnn-htp: init IO tensor using {}", m_graph_map.begin()->first);
  if (true != m_ioTensor->initialize(m_graph_map.begin()->second->context_handle)) {
    __ERROR("qnn-htp: failure to initialize IOTensor");
    return false;
  }

  // Technical note: unordered_map is faster thans map but map makes debug logs easier to read
  // The runtime impact shouldn't be very large since max size < #tensors

  typedef int CtxBitVector;
  // Maps context bitVector to a map{tensor_name -> max_tensor_size}
  std::map<CtxBitVector, std::map<std::string, size_t>> ctx_alloc_map;
  // Maps tensor_name to context bitVector, each bit representing a context the tensor exists in
  std::map<std::string, CtxBitVector> tensor_ctx_map;
  // Maps a ContextHandle to a one-hot encoded bitVector (e.g. 1, 2, 4, ...)
  std::map<Qnn_ContextHandle_t, CtxBitVector> ctx_to_hash;

  // Iterate over all tensors in all GraphVariants to figure out allocations
  for (auto& variant : m_variant_list) {
    // Map the context handle to a hashed bitVector
    if (!ctx_to_hash.contains(variant.context_handle)) {
      ctx_to_hash[variant.context_handle] = 1 << ctx_to_hash.size();
    }
    for (auto& tensor_specs : {variant.input_specs, variant.output_specs}) {
      for (auto& [tname, tspec] : tensor_specs) {
        size_t size           = tspec.dims.getAlignedSize();
        CtxBitVector tcontext = ctx_to_hash[variant.context_handle];

        // Check if it's LoRA enabled model
        if (!_lora_enabled && tname.find("lora") != std::string::npos) _lora_enabled = true;

        if (tensor_ctx_map.contains(tname)) {  // For duplicate tensor names, link them
          CtxBitVector context_bitvec = tensor_ctx_map.at(tname);
          size                        = std::max(ctx_alloc_map[context_bitvec][tname], size);
          if ((context_bitvec & tcontext) == 0)  // Set of contexts needs to be updated
            ctx_alloc_map[context_bitvec].erase(tname);

          tcontext |= context_bitvec;
        }

        ctx_alloc_map[tcontext][tname] = size;
        tensor_ctx_map[tname]          = tcontext;
      }
    }

    // Cleanup is essential in case of very large number of splits
    for (auto it = ctx_alloc_map.cbegin(); it != ctx_alloc_map.cend();)
      it = (it->second.empty()) ? ctx_alloc_map.erase(it) : ++it;
  }

  for (auto& [tcontext, tensor_alloc_map] : ctx_alloc_map) {
    __DEBUG("qnn-htp: ctx_alloc_map[{}] = {{", tcontext);
    for (auto& [tname, tsize] : tensor_alloc_map) __DEBUG("\t{} : {},", tname, tsize);
    __DEBUG("}}");
  }

  // Calculate total allocation sizes and offset of each tensor within its allocated buffer
  if (m_ioTensor->allocateBuffers(ctx_alloc_map, tensor_alloc_info) == false) return false;

  __DEBUG("tensor_alloc_info = \{{");
  for (auto& [tname, toffset] : tensor_alloc_info)
    __DEBUG("\t{}: [{}, {}],", tname, toffset.first, toffset.second);
  __DEBUG("}}");

  // Link the tensor allocs to each nsp graph
  for (auto& graph : m_nsp_graphs) graph.tensor_alloc_info = &tensor_alloc_info;
  // For each variant, map tensor name to its allocated buffer, i/o and offset within the buffer
  // TODO: Check why we aren't just looping over all variants here!

  // For each variant, map tensor name to its allocated buffer, i/o and offset within the buffer
  for (auto& graph_variant : m_variant_list) {
    const int32_t variant = graph_variant.n_tokens;

    std::map<std::string, std::tuple<int, size_t, size_t>> graph_allocs;
    for (auto& [tname, tspec] : graph_variant.input_specs) {
      auto& [alloc_idx, offset] = tensor_alloc_info.at(tname);
      graph_allocs[tname]       = {alloc_idx, offset, tspec.dims.getAlignedSize()};
    }

    for (auto& [tname, tspec] : graph_variant.output_specs) {
      size_t kv_offset = 0;
      size_t size      = tspec.dims.getAlignedSize();

      auto& [alloc_idx, offset] = tensor_alloc_info.at(tname);
      graph_allocs[tname]       = {alloc_idx, offset + kv_offset, size};
    }

    if (!m_ioTensor->mapFusedBufferOffset(
            graph_variant.graph_info, graph_variant.context_handle, graph_allocs)) {
      __ERROR("Error mapping tensor to allocation buffers");
      return false;
    }
  }

  __DEBUG("qnn-htp: init IO tensors complete : {} usec", start.elapsed_usec());

  return true;
}

static bool checkShape(const std::string& tensor_name,
                       const QnnUtils::Tensor* tensor,
                       int32_t height,
                       int32_t width,
                       int32_t channel,
                       int32_t bitWidth,
                       std::vector<std::tuple<std::string, std::string, std::string>>& errors) {
  if (tensor == nullptr) return true;
  const QnnUtils::Dims& tDims = tensor->dims;

  if ((height == -1 || height == tDims.height) && (width == -1 || width == tDims.width) &&
      (channel == -1 || channel == tDims.channel) && (bitWidth == -1 || bitWidth == tDims.bitWidth))
    return true;

  std::stringstream err_msg;
  err_msg << "Expected [ " << height << ", " << width << ", " << channel << "] "
          << "bitWidth=" << bitWidth << ". Found [ " << tDims.height << ", " << tDims.width << ", "
          << tDims.channel << "] "
          << "bitWidth=" << tDims.bitWidth;

  errors.push_back({"ShapeError", tensor_name, err_msg.str()});
  return false;
}

// Run all validations for the model here so we can exit early
bool QnnNspImageModel::validateModel() {
  std::vector<std::tuple<std::string, std::string, std::string>> errors;

  QnnUtils::Tensor* tt;

  // default input type is token
  m_inputType = InputType::PIXELS;

  // Check 1 - input layer exists
  for (auto& [input_size, variant] : m_nsp_graphs.front().variants) {
    if ((tt = variant->getInput(m_layerNames[LayerType::INPUT])) == nullptr) {
      errors.push_back({variant->graph_name, m_layerNames[LayerType::INPUT], "Tensor not found"});
    } else {
      checkShape(m_layerNames[LayerType::INPUT], tt, -1, -1, -1, tt->dtype.bw(), errors);
    }
  }

  // Check 2 - In case of VEG models :-> check if output layer exists.
  for (auto& [input_size, variant] : m_nsp_graphs.back().variants) {
    if ((tt = variant->getOutput(m_layerNames[LayerType::OUTPUT])) == nullptr)
      errors.push_back({variant->graph_name, m_layerNames[LayerType::OUTPUT], "Tensor not found"});
  }
  return true;
}

inline bool QnnNspImageModel::updateTensorPointer(GraphVariant& variant,
                                                  std::string& key,
                                                  QnnUtils::Tensor*& t) {
  QnnUtils::Tensor* tensor_ptr = variant.getInput(key);
  if (tensor_ptr == nullptr) {
    tensor_ptr = variant.getOutput(key);
    if (tensor_ptr == nullptr) return true;
  }
  if (t == nullptr) t = tensor_ptr;
  getBuffer(tensor_ptr);
  if (getBuffer(t) == getBuffer(tensor_ptr)) return true;
  __ERROR("{} has different addresses: {} vs {}", key, (void*)t, (void*)tensor_ptr);
  return false;
}

bool QnnNspImageModel::initializeTensorPointers() {
  // Ideally this needs to be done for all sets of AR-n available, e.g. for AR-1 and AR-1024

  bool status = true;
  for (auto& variant : m_variant_list) {
    status &= updateTensorPointer(variant, m_layerNames[LayerType::INPUT], t_pixel_values);
    status &= updateTensorPointer(variant, m_layerNames[LayerType::OUTPUT], t_image_features);
  }
  if (!status) __ERROR("qnn-htp: Error in setting up named tensor pointers.");
  status &= !(!t_pixel_values);
  if (!t_pixel_values) __ERROR("Tensor not found: {}", m_layerNames[LayerType::INPUT]);
  status &= !(!t_image_features);
  if (!t_image_features) __ERROR("Tensor not found: {}", m_layerNames[LayerType::OUTPUT]);
  // Detect activation bitwidth
  if (status) {
    // Check Input->
    d_input = t_pixel_values->dtype;
    if (!supported_activations.contains(d_input)) {
      __ERROR("Input Tensor: {} as unsupported activation type {}",
              m_layerNames[LayerType::INPUT],
              d_input.str());
      status = false;
    }
    __DEBUG("qnn-htp datatypes: d_input {} ", d_input.str());

    if (!status) __ERROR("Only 8-bit, 16-bit and 32-bit activations are supported");

    d_output = t_image_features->dtype;
    if (!supported_activations.contains(d_output)) {
      __ERROR("Output Tensor: {} as unsupported activation type {}",
              m_layerNames[LayerType::OUTPUT],
              d_output.str());
      status = false;
    }

    __DEBUG("qnn-htp datatypes: d_output {} ", d_output.str());

    if (!status) __ERROR("Only 8-bit, 16-bit and 32-bit activations are supported");
  }

  return status;
}

template <typename DType>
bool QnnNspImageModel::setupInput(const std::vector<uint8_t>& inputs) {
  // Setup pixel_values tensor
  {
    uint32_t rank      = t_pixel_values->tensor->v1.rank;
    size_t numElements = 1;
    for (int i = 0; i < rank; i++) {
      numElements *= t_pixel_values->tensor->v1.dimensions[i];
    }
    uint32_t bufferSize = d_input.bw() * numElements;
    if (embedding_datatype == "QNN_DATATYPE_FLOAT_32") {
      float* embeddingSrc = (float*)(inputs.data());
      quantizeInput(embeddingSrc, 0, numElements);
    } else {  // native datatype
      // Copy the data input vector
      std::copy(inputs.data(), inputs.data() + bufferSize, (uint8_t*)getBuffer(t_pixel_values));
    }
  }

  return true;
}

bool QnnNspImageModel::setupInputFP16(const std::vector<uint8_t>& inputs) {
  // Placeholder for FP16 inputs
  return true;
}

bool QnnNspImageModel::setupInputTensors(const std::vector<uint8_t>& inputs) {
  qualla::Timer start;

  // clang-format off
    switch (d_input) {
    case QNN_DATATYPE_UFIXED_POINT_8:
        setupInput<uint8_t>(inputs); break;
    case QNN_DATATYPE_UFIXED_POINT_16:
        setupInput<uint16_t>(inputs); break;
    case QNN_DATATYPE_INT_32:
        setupInput<int32_t>(inputs); break;
    case QNN_DATATYPE_FLOAT_16: {
        setupInputFP16(inputs);
        break;
    }
    default: __ERROR("Unsupported attention mask dtype {}", d_input.str()); return false;
    }
  // clang-format on

  __TRACE("qnn-htp: setup-input-tensors complete : {} usec", start.elapsed_usec());
  return true;
}

size_t QnnNspImageModel::runInference(const std::vector<uint8_t>& inputs,
                                      std::vector<uint8_t>& outputs) {
  qualla::Timer start;

  // Select variant based on variant_latency, or default to current variant
  std::vector<uint8_t> pixel_inputs(inputs);

  // If variant selected in BERT-Mode, append token history to current request
  int32_t variant = 0;

  // Technical note: int32_t can hold upto 596 hours
  // Even int16_t should be sufficient here - it holds upto 32.8 seconds
  int32_t total_wait = 0;
  int32_t total_exec = 0;

  if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
    if (!setupInputTensors(inputs)) return 0;

    for (auto& nsp_graph : m_nsp_graphs) {
      //__DEBUG("execute({}, {}, {})", variant, m_inference_count, wait_kv_update_count);
      if (!nsp_graph.execute(0, m_inference_count, 0, graph_switching, lazy_lora)) return false;
    }

    m_inference_count++;
  }
  uint32_t rank      = t_image_features->tensor->v1.rank;
  size_t numElements = 1;
  for (int i = 0; i < rank; i++) {
    numElements *= t_image_features->tensor->v1.dimensions[i];
  }

  // outputs.resize(numElements * t_image_features->dtype.bw());
  uint8_t* output_buffer = (uint8_t*)getBuffer(t_image_features);
  outputs.insert(
      outputs.begin(), output_buffer, output_buffer + numElements * t_image_features->dtype.bw());
  __DEBUG("qnn-htp: run-inference complete : {} usec : wait {} exec {}",
          start.elapsed_usec(),
          total_wait,
          total_exec);

  // threadpool.suspend();
  return 1;
}

bool QnnNspImageModel::quantizeInput(float* in, size_t tensorOffset, size_t length) {
  if (t_pixel_values == nullptr) {
    __ERROR("Input Tensor {} not found during execute", m_layerNames[LayerType::INPUT]);
    return false;
  }

  const auto scale  = t_pixel_values->quantParam[0].scale;
  const auto offset = t_pixel_values->quantParam[0].offset;
  // clang-format off
    switch (t_pixel_values->dtype) {
        case QNN_DATATYPE_UFIXED_POINT_8: QnnUtils::quantizeTensorPtr(in, (uint8_t*)getBuffer(t_pixel_values) + tensorOffset, offset, scale, length); break;
        case QNN_DATATYPE_UFIXED_POINT_16: QnnUtils::quantizeTensorPtr(in, (uint16_t*)getBuffer(t_pixel_values) + tensorOffset, offset, scale, length); break;
        default: __ERROR("Unsupported alpha tensor dtype {}", t_pixel_values->dtype.str()); return false;
    }
    return true;
}

void QnnNspImageModel::getTensorDimensions(LayerType layerType,
                                   std::vector<uint32_t>& dimensions)
{
  if (layerType == LayerType::OUTPUT) {
    dimensions.push_back(t_image_features->dims.height);
    dimensions.push_back(t_image_features->dims.width);
    dimensions.push_back(t_image_features->dims.channel);
  }
}

void QnnNspImageModel::getTensorParam(LayerType layerType,
                              std::string& dataType,
                              double& scale,
                              int32_t& offset,
                              size_t& bitWidth)
{
  if (layerType == LayerType::OUTPUT) {
    dataType = t_image_features->dtype.str();
    scale    = t_image_features->quantParam[0].scale;
    offset   = t_image_features->quantParam[0].offset;
    bitWidth = t_image_features->dtype.bw();
  }
}

size_t QnnNspImageModel::getEmbeddingBufferSize() {
    return 0;
}

void  QnnNspImageModel::setHigherVariant(){
    return;
}


} // namespace qualla
