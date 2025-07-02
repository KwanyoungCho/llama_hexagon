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
#include "native-kv.hpp"
#include "nsp-model.hpp"
#include "qualla/detail/cache-file.hpp"
#include "qualla/detail/timer.hpp"
#include "qualla/env.hpp"
#include "smart-mask.hpp"

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

namespace fs = std::filesystem;

namespace qualla {

QnnNspModel::QnnNspModel(Env& env, const QnnNspBaseModel::Params& params)
    : QnnNspBaseModel(env, params) {
  // Initialize QnnAPI
  m_qnnApi = std::unique_ptr<QnnApi>(new QnnApi());

  spill_fill_buffer_size  = params.spill_fill_bufsize;
  m_kv_dim                = params.kv_dim;
  m_use_mmap              = params.use_mmap;
  m_use_async_Init        = params.use_async_Init;
  mmap_budget             = params.mmap_budget;
  m_dataAlignmentSize     = params.data_alignment_size;
  m_ctx_size              = params.ctx_size;
  m_pad_token             = params.pad_token;
  lmhead_weight_dir       = params.lmhead_weight_dir;
  graph_switching         = params.graph_switching;
  lazy_lora               = params.lazy_lora;
  load_select_graphs      = params.load_select_graphs;
  embedding_length        = params.embedding_length;
  embedding_datatype      = params.embedding_datatype;
  m_disableKvCache        = params.disable_kv_cache;
  m_embd_size             = params.n_embd;
  m_modelArchitectureType = params.modelArchitectureType;
  // Positional encoding parameters
  m_positional_encoding = params.positional_encoding_params;
  if (m_positional_encoding.type == PositionalEncoding::ROPE)  // Save m_pos_dim for easy access
    m_pos_dim = m_positional_encoding.rope_params.dims;

  // Longcontext params
  m_longcontext = params.longcontext_params;

  m_draft_tok_map = params.draft_tok_map;
  if (graph_switching && !m_use_mmap)
    __WARN("Graph switching with non-mmaped implementation can cause high sustained memory usage");

  variant_latency = params.variant_latency;

  if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
    m_pooled_output = params.pooled_output;
  }

  exec_select_graphs = params.exec_select_graphs;
  if (!exec_select_graphs.empty())
    __DEBUG("qnn-htp : Execute selected graphs = {}", exec_select_graphs);

  if (params.kv_update_method == "SHIFT_CONCAT" || (params.kv_update_method == "POINTER_SHIFT"))
    __WARN("kv-update-method is deprecated. Defaulting to SMART_MASK or NATIVE_KV");
  _kv_update_method = SMART_MASK;  // Updates to NATIVE_KV if HMX_WEIGHT_LAYOUT tensor is found

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

  // Initialize QNN IO Tensor
  m_ioTensor = std::unique_ptr<IOTensor>(
      new IOTensor(m_sharedBuffer ? BufferAlloc::SHARED_BUFFER : BufferAlloc::DEFAULT,
                   m_sharedBuffer ? m_qnnApi->getQnnInterfaceVer() : nullptr));

  m_qnnApi->setIOTensorBufferMgr(m_ioTensor.get());
  m_qnnApi->setKVDim(m_kv_dim);
  m_qnnApi->setContextSize(m_ctx_size);
  m_qnnApi->setKVUpdateMethod(_kv_update_method);

  if (params.debug_specs || params.debug_tensors) {
    if (!fs::exists(params.debug_path) && !fs::create_directories(params.debug_path))
      throw std::runtime_error("Could not create debug directory : " + params.debug_path);
  }
}

QnnNspModel::~QnnNspModel() {
  qualla::Timer start;

  // The threadpool needs to be stopped before KVManager
  // destruction to avoid race conditions.
  m_kvmanager->stopThreadpool();

  // Free cached RoPE memory
  if (rope_sin != nullptr) free(rope_sin);
  if (rope_cos != nullptr) free(rope_cos);

  if (eagle_extra_feature != nullptr) {
    free(eagle_extra_feature);
    eagle_extra_feature = nullptr;
  }
  _counter = nullptr;
  __DEBUG("qnn-htp: model destruct complete: {} usec", start.elapsed_usec());
}

// Given a filename, initializeModel load and initializes QNN runtime libraries and the model
bool QnnNspModel::initializeModel(void) {
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

  if (_debug_specs) dumpTensorSpecs();

  // Compile the number of LLM graphs and auxiliary graphs
  const auto m_num_graphs = m_qnnApi->getGraphsCount();
  const auto graphs_info  = m_qnnApi->getGraphsInfo();

  __INFO("qnn-api initialized with {} graph(s)", m_num_graphs);

  m_variant_list.reserve(m_num_graphs);
  std::map<std::pair<int32_t, int32_t>, std::set<std::string>> graph_names;
  for (size_t graph_idx = 0; graph_idx < m_num_graphs; graph_idx++) {
    qnn_wrapper_api::GraphInfo_t* const graph_info = graphs_info[graph_idx];
    const std::string graph_name                   = std::string(graph_info->graphName);

    __DEBUG("qnn-htp: Graph {}", graph_name);
    GraphVariant graph(graph_info, m_qnnApi->getContexts(graph_info), m_layerNames, _env);

    if (!variant_latency.empty() && !variant_latency.contains(graph.n_tokens)) {
      __WARN("qnn-htp: Disabling {} based on conf file", graph_name);
      continue;
    }

    if (exec_select_graphs.size() != 0 &&
        std::find(exec_select_graphs.begin(), exec_select_graphs.end(), graph_name) ==
            exec_select_graphs.end()) {
      __DEBUG("qnn-htp: Graph {} is not selected to execute based on conf file", graph_name);
      continue;
    }

    m_variant_list.emplace_back(graph);
    m_graph_map[graph_name] = &m_variant_list.back();

    std::pair<int32_t, int32_t> variant_spec = {graph.n_tokens, graph.ctx_size};
    nsp_graph_count[variant_spec]++;
    graph_names[variant_spec].insert(graph_name);
  }

  // Collect all available ctx_sizes so we can handle not being able to detect ctx_size in a variant
  std::unordered_set<int32_t> available_ctx_size;
  for (const auto& [variant_spec, count] : nsp_graph_count) {
    if (variant_spec.second != -1) available_ctx_size.insert(variant_spec.second);
  }

  std::vector<std::pair<int32_t, int32_t>> keysToDelete;
  // For all variants where we did not detect a ctx_size, add it to all ctx_sizes
  for (const auto& [variant_spec, count] : nsp_graph_count) {
    if (variant_spec.second != -1) continue;
    auto& prev_names = graph_names.at(variant_spec);
    for (const auto& new_ctx : available_ctx_size) {
      std::pair<int32_t, int32_t> new_spec = {variant_spec.first, new_ctx};
      nsp_graph_count[new_spec]++;
      auto& new_names = graph_names[new_spec];
      new_names.insert(prev_names.begin(), prev_names.end());
    }
    keysToDelete.push_back(variant_spec);
  }

  for (const auto& key : keysToDelete) {
    graph_names.erase(key);
    nsp_graph_count.erase(key);
  }

  if (exec_select_graphs.size() != 0 && graph_names.empty()) {
    __ERROR("No matching graphs based on conf file");
  }

  // Create NSPGraph for each splits
  int32_t n_splits = 0;
  for (auto& [_, count] : nsp_graph_count) n_splits = std::max(n_splits, count);
  m_nsp_graphs.reserve(n_splits);
  for (int idx = 0; idx < n_splits; idx++) {
    m_nsp_graphs.emplace_back(idx, _env, m_qnnApi.get(), m_ioTensor.get());
    m_nsp_graphs.back().setDebugMode(_debug_specs, _debug_tensors, _debug_path);
  }

  // Insert all GraphVariants into corresponding NSPGraph
  for (auto& [variant_spec, graphs] : graph_names) {
    const auto& [variant, ctx_size] = variant_spec;
    int idx = 0;  // Graph names are sorted by default (std::set<>), so iterate by split
    for (auto& graph_name : graphs) {
      __INFO("Inserting graph {} as idx {} for AR-{} CL-{}", graph_name, idx, variant, ctx_size);
      m_nsp_graphs[idx++].addGraph(m_graph_map.at(graph_name));
    }
  }

  // Detect whether NATIVE_KV needs to be activated
  for (auto& variant : m_variant_list) {
    for (auto& [tname, tspec] : variant.output_specs) {
      // If QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT is detected, we switch to NATIVE_KV format
      if (tspec.tensor->v1.dataFormat == QNN_TENSOR_DATA_FORMAT_HMX_WEIGHT_LAYOUT) {
        _kv_update_method    = NATIVE_KV;
        m_expectedDataFormat = tspec.tensor->v1.dataFormat;
        m_qnnApi->setKVUpdateMethod(_kv_update_method);
        break;
      }
    }
    if (_kv_update_method == NATIVE_KV) break;
  }

  __INFO("qnn-htp: Graphs loaded ((AR-n, CL-x): #splits): {}", nsp_graph_count);

  int32_t max_ctx_size = 0;
  for (auto& [variant_spec, count] : nsp_graph_count)
    max_ctx_size = std::max(max_ctx_size, variant_spec.second);
  if (max_ctx_size < m_ctx_size && m_longcontext.mode == LongContextParams::DISABLED) {
    // If LongContext is disabled, make sure the config CL matches loaded CL
    State::error(fmt::format(
        "Config specifies context->size={}, but loaded max-CL={}", m_ctx_size, max_ctx_size));
    return false;
  }

  __DEBUG("qnn-htp: Model Init complete: {} usec", start.elapsed_usec());
  return true;
}

// Once the model has been loaded, initialize IO Tensors
// m_ioTensors is initialized by the context for now
bool QnnNspModel::initializeIOTensors() {
  if (m_use_async_Init == false) {
    // IO Tensor Mem Registration is already done within the
    // model_initailize by Qnn_API for Sync Init.

    // set lmHeadWeightsEnabled and loraWeights Enabled
    _lmhead_weight_input = m_qnnApi->getLmHeadWeightInputEnabled();
    _lora_enabled        = m_qnnApi->getLoraWeightEnabled();
    for (auto it = nsp_graph_count.rbegin(); it != nsp_graph_count.rend(); ++it) {
      for (QnnNspGraph& graph : m_nsp_graphs) {
        // TensorAllocInfo is added to each NSP graph.
        // Needed by Pointer_SHIFT Registration During Execute.
        graph.tensor_alloc_info = m_qnnApi->getTensorAllocInfo();
        if (graph.tensor_alloc_info == NULL) {
          __ERROR("Error Tensor Allocation Failed.");
          return false;
        }
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
  if (true !=
      m_ioTensor->initialize(m_graph_map.begin()->second->context_handle, m_dataAlignmentSize)) {
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
        // Check if graph has lmhead weight input
        if (!_lmhead_weight_input && tname.compare("weight") == 0) _lmhead_weight_input = true;

        // Allocate KV Tensors as in+out
        if (tname.starts_with("past_")) {
          if (tname.ends_with("_in")) continue;  // kv_in is processed along with kv_out

          // For kv_out, add the size of kv_in as well
          const std::string tname_in = tname.substr(0, tname.rfind('_')).append("_in");
          if (auto tensor = variant.getInput(tname_in)) size += tensor->dims.getAlignedSize();

          d_kv = QnnUtils::DataType(tspec.tensor);

          // Allocate extra buffer for pointer shift
          // 1024-n for keys (1024-n)*128 for values
          // For aligned size, we might as well use 1024 and 128*1024
          if (_kv_update_method == POINTER_SHIFT)
            size += (tname.starts_with("past_key")) ? variant.ctx_size * d_kv.bw()
                                                    : variant.ctx_size * m_kv_dim * d_kv.bw();
        }

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
  for (auto& graph_variant : m_variant_list) {
    const int32_t ctx_size = graph_variant.ctx_size;

    std::map<std::string, std::tuple<int, size_t, size_t>> graph_allocs;
    for (auto& [tname, tspec] : graph_variant.input_specs) {
      if (tname.starts_with("past_")) continue;
      auto& [alloc_idx, offset] = tensor_alloc_info.at(tname);
      graph_allocs[tname]       = {alloc_idx, offset, tspec.dims.getAlignedSize()};
    }

    for (auto& [tname, tspec] : graph_variant.output_specs) {
      size_t kv_offset = 0;
      size_t size      = tspec.dims.getAlignedSize();

      auto& [alloc_idx, offset] = tensor_alloc_info.at(tname);
      if (tname.starts_with("past_")) {
        auto in_name = tname.substr(0, tname.rfind("_")).append("_in");
        if (auto kv_in = graph_variant.getInput(in_name)) {
          kv_offset = kv_in->dims.getAlignedSize();
          if (_kv_update_method == POINTER_SHIFT)
            kv_offset += (tname.starts_with("past_key")) ? ctx_size * d_kv.bw()
                                                         : ctx_size * m_kv_dim * d_kv.bw();
          graph_allocs[in_name] = {alloc_idx, offset, kv_offset};
        }
      }
      graph_allocs[tname] = {alloc_idx, offset + kv_offset, size};
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

/* Converts "don't care" dimensions into "*" */
static std::string translateDim(int32_t dim) { return (dim == -1) ? "*" : std::to_string(dim); }

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
  err_msg << "Expected [ " << translateDim(height) << ", " << translateDim(width) << ", "
          << translateDim(channel) << "] "
          << "bitWidth=" << translateDim(bitWidth) << ". Found [ " << tDims.height << ", "
          << tDims.width << ", " << tDims.channel << "] "
          << "bitWidth=" << tDims.bitWidth;

  errors.push_back({"ShapeError", tensor_name, err_msg.str()});
  return false;
}

// Run all validations for the model here so we can exit early
bool QnnNspModel::validateModel() {
  // Checks we will be running
  // 1a. input_ids or inputs_embeds exists in the first split
  // 1b. token_type_ids should exists in case of Bert
  // 2. logits exists in the last split
  // 3. Shapes for all named tensors are correct
  // 4. All tensors with identical names (incl kv_in/kv_out) have identical quantization params
  // Missing check : Shape of tensor between splits match up

  // Support for 16-bit KV Tensors is temporarily disabled
  // If you need this, please refer to past commits (QuaLLA <= v0.3.22)

  // Important : These variables need to be set correctly
  // m_vocab_size  - Calculated as max(logits.shape) since len()
  // m_kv_dim      - Calculated in this function before usage
  // m_ctx_size    - Provided by the user as n_ctx
  std::vector<std::tuple<std::string, std::string, std::string>> errors;

  QnnUtils::Tensor* tt;

  // default input type is token
  m_inputType = InputType::TOKENS;

  // Check 1 - input layer exists
  for (auto& [variant_spec, variant] : m_nsp_graphs.front().variants) {
    const auto& [n_tokens, ctx_size] = variant_spec;
    // Update model expectations for E2T if an inputs_embeds layer is present. marks the input Type
    if ((tt = variant->getInput("inputs_embeds")) != nullptr) {
      m_layerNames[LayerType::INPUT] = "inputs_embeds";
      m_inputType                    = InputType::EMBEDDINGS;
    } else if ((tt = variant->getInput("_model_embed_tokens_Gather_Gather_output_0")) != nullptr) {
      // workaround to support split LLM (LUT + Decoder)
      m_layerNames[LayerType::INPUT] = "_model_embed_tokens_Gather_Gather_output_0";
      m_inputType                    = InputType::EMBEDDINGS;
    } else if ((tt = variant->getInput("_model_model_embed_tokens_Gather_Gather_output_0")) !=
               nullptr) {
      // workaround to support split LLM (LUT + Decoder)
      m_layerNames[LayerType::INPUT] = "_model_model_embed_tokens_Gather_Gather_output_0";
      m_inputType                    = InputType::EMBEDDINGS;
    } else if ((tt = variant->getInput("_model_embedding_concat_Concat_Concat_output_0")) !=
               nullptr) {
      // workaround to support split LLM (LUT + Decoder)
      m_layerNames[LayerType::INPUT] = "_model_embedding_concat_Concat_Concat_output_0";
      m_inputType                    = InputType::EMBEDDINGS;
    }
    if ((tt = variant->getInput(m_layerNames[LayerType::INPUT])) == nullptr) {
      errors.push_back({variant->graph_name, m_layerNames[LayerType::INPUT], "Tensor not found"});
    } else {
      input_bitWidth = tt->dtype.bw();
      checkShape(m_layerNames[LayerType::INPUT], tt, -1, -1, -1, input_bitWidth, errors);

      if (embedding_datatype == "QNN_DATATYPE_FLOAT_32") {
        m_embeddingBufferSize = m_embd_size * sizeof(float);
      } else {
        m_embeddingBufferSize = m_embd_size * input_bitWidth;
      }

      // For embedding inputs, the expected count is multiplied by the embedding size.
      size_t expectedElementCount =
          (m_inputType == InputType::TOKENS) ? n_tokens : n_tokens * m_embd_size;
      if (m_layerNames[LayerType::INPUT] == "_model_embedding_concat_Concat_Concat_output_0") {
        expectedElementCount = expectedElementCount * 2;
      }
      if (tt->dims.getNumElements() != expectedElementCount)
        errors.push_back(
            {variant->graph_name, m_layerNames[LayerType::INPUT], "Wrong input shape"});
    }
  }

  // Check 1b - In case of BERT :-> token_type_ids
  if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
    for (auto& [variant_spec, variant] : m_nsp_graphs.front().variants) {
      const auto& [n_tokens, ctx_size] = variant_spec;
      if ((tt = variant->getInput(m_layerNames[LayerType::TOKEN_TYPE_IDS])) == nullptr)
        errors.push_back(
            {variant->graph_name, m_layerNames[LayerType::TOKEN_TYPE_IDS], "Tensor not found"});
      else {
        checkShape(m_layerNames[LayerType::TOKEN_TYPE_IDS], tt, -1, -1, -1, 4, errors);
        if (tt->dims.getNumElements() != n_tokens)
          errors.push_back({variant->graph_name,
                            m_layerNames[LayerType::TOKEN_TYPE_IDS],
                            "Wrong token_type_ids shape"});
      }
    }
  }

  // Check 2 - In case of LLama :-> logits exists
  //           In case of BERT :-> pooled_output & sequence_outputs exists
  for (auto& [variant_spec, variant] : m_nsp_graphs.back().variants) {
    const auto& [n_tokens, ctx_size] = variant_spec;
    if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
      if ((tt = variant->getOutput(m_layerNames[LayerType::POOL_OUTPUT])) == nullptr)
        errors.push_back(
            {variant->graph_name, m_layerNames[LayerType::POOL_OUTPUT], "Tensor not found"});
      else {
        if (tt->dims.getNumElements() != m_embd_size)
          errors.push_back({variant->graph_name,
                            m_layerNames[LayerType::POOL_OUTPUT],
                            "Wrong pooled_outputs shape"});
      }
      if (!m_pooled_output) {
        if ((tt = variant->getOutput(m_layerNames[LayerType::SEQ_OUTPUT])) == nullptr)
          errors.push_back(
              {variant->graph_name, m_layerNames[LayerType::SEQ_OUTPUT], "Tensor not found"});
        else {
          if (tt->dims.getNumElements() != n_tokens * m_embd_size)
            errors.push_back({variant->graph_name,
                              m_layerNames[LayerType::SEQ_OUTPUT],
                              "Wrong sequence_output shape"});
        }
      }
    } else {
      if ((tt = variant->getOutput(m_layerNames[LayerType::OUTPUT])) == nullptr)
        errors.push_back(
            {variant->graph_name, m_layerNames[LayerType::OUTPUT], "Tensor not found"});
      else {
        if (m_vocab_size == -1) m_vocab_size = tt->dims.getMaxDim();
        if (tt->dims.getNumElements() != m_vocab_size &&
            tt->dims.getNumElements() != n_tokens * m_vocab_size)
          errors.push_back(
              {variant->graph_name, m_layerNames[LayerType::OUTPUT], "Wrong logits shape"});
      }
    }
  }

  // Check 3 - Shapes for all names tensors are correct
  if (m_kv_dim == -1) {  // Deduce KV$ embed_dim if not already available
    for (auto& variant : m_variant_list) {
      for (auto& [tname, tspec] : variant.output_specs)
        if (tname.starts_with("past_key")) m_kv_dim = tspec.dims.width;
      if (m_kv_dim != -1) break;
    }
  }

  // Detect if the model uses Scatter (new_key -> past_key) or Concat (past_key + new_key)
  bool scatter_detect_flag = false;
  for (auto& variant : m_variant_list) {
    for (auto& [tname, tspec] : variant.input_specs)
      if (tname.starts_with("past_key")) {
        m_kv_use_scatter    = tspec.dims.channel == variant.ctx_size;
        scatter_detect_flag = true;
        break;
      }
    if (scatter_detect_flag) break;
  }

  for (auto& variant : m_variant_list) {
    const int32_t n_tokens = variant.n_tokens;
    const int32_t ctx_size = variant.ctx_size;

    // Verify attention mask tensors
    if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
      tt = variant.getInput(m_layerNames[LayerType::ATTN_MASK]);
      checkShape(m_layerNames[LayerType::ATTN_MASK], tt, 1, 1, ctx_size, -1, errors);
    } else {
      tt = variant.getInput(m_layerNames[LayerType::ATTN_MASK]);
      checkShape(m_layerNames[LayerType::ATTN_MASK], tt, 1, n_tokens, ctx_size, -1, errors);
    }

    // Verify positional encoding tensors
    if (m_positional_encoding.type == PositionalEncoding::ROPE) {
      tt = variant.getInput(m_layerNames[LayerType::POS_SIN]);
      checkShape(m_layerNames[LayerType::POS_SIN], tt, 1, n_tokens, m_pos_dim, -1, errors);
      tt = variant.getInput(m_layerNames[LayerType::POS_COS]);
      checkShape(m_layerNames[LayerType::POS_COS], tt, 1, n_tokens, m_pos_dim, -1, errors);
    } else if (m_positional_encoding.type == PositionalEncoding::ABSOLUTE) {
      tt = variant.getInput(m_layerNames[LayerType::POS_IDS]);
      checkShape(m_layerNames[LayerType::POS_IDS], tt, 1, 1, n_tokens, -1, errors);
    } else if (m_positional_encoding.type == PositionalEncoding::ALIBI) {
      tt = variant.getInput(m_layerNames[LayerType::POS_IDS]);
      checkShape(m_layerNames[LayerType::POS_IDS], tt, 1, n_tokens, ctx_size, -1, errors);
    }

    // Verify KV$ tensors
    if (m_modelArchitectureType != ModelArchitectureType::ENCODER) {
      for (auto& [tname, tspec] : variant.input_specs) {
        const int32_t past_dim = m_kv_use_scatter ? ctx_size : ctx_size - n_tokens;
        if (tname.starts_with("past_key"))
          checkShape(tname, &tspec, -1, m_kv_dim, past_dim, -1, errors);
        else if (tname.starts_with("past_value"))
          checkShape(tname, &tspec, -1, past_dim, m_kv_dim, -1, errors);
      }

      for (auto& [tname, tspec] : variant.output_specs) {
        if (tname.starts_with("past_key"))
          checkShape(tname, &tspec, -1, m_kv_dim, n_tokens, -1, errors);
        else if (tname.starts_with("past_value"))
          checkShape(tname, &tspec, -1, n_tokens, m_kv_dim, -1, errors);
      }
    }
  }

  // skip check in case of BERT architecture since no KV cache tensors are existing
  if (m_modelArchitectureType != ModelArchitectureType::ENCODER) {
    // Check 4 - Quantization parameter match
    std::unordered_map<std::string, QnnUtils::QuantParam> quant_params;
    for (auto& variant : m_variant_list) {
      for (auto& tensor_specs : {variant.input_specs, variant.output_specs}) {
        for (auto& [tname, tspec] : tensor_specs) {
          std::string name = (tname.starts_with("past_") && tname.ends_with("_in"))
                                 ? tname.substr(0, tname.rfind("_")).append("_out")
                                 : tname;
          if (name.compare(m_layerNames[LayerType::OUTPUT]) == 0) continue;
          if (quant_params.contains(name)) {
            if (quant_params.at(name).scale != tspec.quantParam[0].scale ||
                quant_params.at(name).offset != tspec.quantParam[0].offset) {
              errors.push_back({variant.graph_name,
                                tname,
                                "Non-identical quantization parameters found for the same tensor"});
            }
          } else {
            quant_params[tname] = {tspec.quantParam[0].scale, tspec.quantParam[0].offset};
          }
        }
      }
    }
  }

  if (errors.size() > 0) {
    QNN_ERROR("Model Validation Errors found");
    for (auto& [graph_name, tensor_name, err_msg] : errors)  // Log the list of errors
      QNN_ERROR("%s : %s - %s", graph_name.c_str(), tensor_name.c_str(), err_msg.c_str());
    QNN_ERROR("Note: Dimensions denoted by '%s' are ignored (i.e. no comparison)",
              translateDim(-1).c_str());
    QNN_ERROR("Check model i/o specs (set dump-specs=true in config) for debugging");
    State::fatal("Error validating HTP models");
    return false;
  }

  return true;
}

bool QnnNspModel::initializeKVManager(const size_t numThreads,
                                      const uint64_t cpuMask,
                                      const bool enablePolling) {
  if (_kv_update_method == NATIVE_KV) {
    m_kvmanager = std::make_unique<NativeKV>(_env, numThreads, cpuMask, enablePolling);
  } else {
    m_kvmanager = std::make_unique<SmartMask>(_env, numThreads, cpuMask, enablePolling);
  }
  __DEBUG("Initializing with KV$ update method = {}", getManagerModeStr(_kv_update_method));

  // Register datatypes. Based on QnnTypes.h, floating types have type 0x02xx
  m_kvmanager->registerDataType(d_kv.bw(), d_kv.type() != 2);

  // Register supported variants
  for (auto& [_, variant] : m_nsp_graphs.front().variants)
    m_kvmanager->registerSupportedVariant(variant->n_tokens, variant->ctx_size);

  // Pick largest variant/context size. This is not important for tensor mapping since
  // all buffers link to the same address anyway, but it will be important for scorer validation.
  const auto [n_tokens, ctx_size] = nsp_graph_count.rbegin()->first;

  // Extract KV$ tensors from model I/O specs
  int32_t anchor_bitwidth = 0;  // Detect bitwidth for anchor in/out tensor for KeyDiff (0=disabled)
  std::map<uint32_t, std::array<std::tuple<int, size_t>, 2>> scorer_allocs;
  std::map<int, std::vector<KVTensor>> kv_tensors;  // maps graph index to set of KV Tensors
  for (auto& graph : m_nsp_graphs) {
    GraphVariant* variant = graph(n_tokens, ctx_size);

    // Parse KV$ Tensor names here - supports past_{key,value}_{layer_idx}[_h{head_idx}]_{in,out}
    // Output type 0 - past_key_{layer_idx}[_h{head_idx}]_{in,out}
    // Output type 1 - past_value_{layer_idx}[_h{head_idx}]_{in,out}
    // Output type 2 - anchor_buffer_{layer_idx}[_h{head_idx}]_in
    // Output type 3 - anchor_buffer_{layer_idx}[_h{head_idx}]_out

    // Accumulate key and value cache tensors (ordered by layer/head)
    std::map<uint32_t, QnnUtils::Tensor* [4]> qnn_kv_map;
    for (auto& [tname, tensor] : variant->output_specs) {
      const bool is_key    = tname.starts_with("past_key");
      const bool is_val    = tname.starts_with("past_value");
      const bool is_anchor = tname.starts_with(m_layerNames.at(LayerType::ANCHOR));
      if (!is_key && !is_val && !is_anchor) continue;

      // Grab the corresponding input tensor if available
      std::string in_tname        = tname.substr(0, tname.rfind("_")).append("_in");
      QnnUtils::Tensor* in_tensor = variant->getInput(in_tname);
      const auto parsed = QnnUtils::parseNumberFromString<2>(tname);  // Get [layer_idx, head_idx]
      const uint32_t index = parsed[0] << 16 | parsed[1];

      if (is_key || is_val) {  // For KV$ add input tensor if available, else output tensor
        qnn_kv_map[index][is_key ? 0 : 1] = (in_tensor == nullptr) ? &tensor : in_tensor;
        if (is_key) scorer_allocs[index][1] = graph.tensor_alloc_info->at(tname);
      } else {
        anchor_bitwidth         = in_tensor->dims.bitWidth;
        qnn_kv_map[index][2]    = in_tensor;
        qnn_kv_map[index][3]    = &tensor;
        scorer_allocs[index][0] = graph.tensor_alloc_info->at(in_tname);
      }
    }

    // Check that each Key has a corresponding Value and vice-versa
    for (const auto& [idx, qnn_kv] : qnn_kv_map) {
      if (qnn_kv[0] && qnn_kv[1]) continue;  // Found both Key and Value Tensor

      // Construct a nice error message to alert about the missing tensor
      uint16_t layer_idx = idx >> 16, head_idx = idx & 0xffff;
      std::string err_msg = fmt::format("Error in layer {} ", layer_idx);
      if (head_idx) err_msg += fmt::format("head {} ", head_idx);
      if (qnn_kv[0] == nullptr) err_msg += "Found Value but no Key Tensor";
      if (qnn_kv[1] == nullptr) err_msg += "Found Key but no Value Tensor";
      State::error(err_msg);
      return false;
    }

    const size_t num_kv = qnn_kv_map.size();
    if (num_kv == 0) continue;  // No KV$ tensors found in the graph

    for (auto& [idx, qnn_kv] : qnn_kv_map) {
      kv_tensors[graph.idx()].emplace_back(
          idx,
          (uint8_t*)getBuffer(qnn_kv[0]),                            // past_key buffer
          (uint8_t*)getBuffer(qnn_kv[1]),                            // past_value buffer
          (int32_t)qnn_kv[0]->dims.height,                           // n_heads
          qnn_kv[0]->quantParam[0],                                  // Quant param for Key
          qnn_kv[1]->quantParam[0],                                  // Quant param for Value
          (!!qnn_kv[2]) ? -qnn_kv[2]->quantParam[0].offset : 0,      // KeyDiff - anchor_offset
          (!!qnn_kv[2]) ? (uint8_t*)getBuffer(qnn_kv[2]) : nullptr,  // KeyDiff - anchor_in
          (!!qnn_kv[3]) ? (uint8_t*)getBuffer(qnn_kv[3]) : nullptr,  // KeyDiff - anchor_out
          nullptr);                                                  // KeyDiff - score buffer
    }
  }

  if (m_longcontext.mode == LongContextParams::KEYDIFF) {
    // Initialize all tensors for the scorer model (anchor/keys/scores)
    // Also add the score buffer pointer associated with each
    std::string scorer_path = (model_basedir / fs::path(m_longcontext.scoring_network)).string();
    __DEBUG("Initializing KeyDiff Scorer {}", scorer_path);
    std::map<uint32_t, uint8_t*> score_memptr;
    if (!m_qnnApi->initializeScorer(
            scorer_path, scorer_allocs, score_memptr, ctx_size, m_expectedDataFormat)) {
      State::error("Failed to initialize scorer");
      return false;
    }
    for (auto& [graph_idx, tensors] : kv_tensors)
      for (auto& tensor : tensors) tensor.scores = score_memptr.at(tensor.idx);

    m_kvmanager->registerQnnApi(m_qnnApi.get());
  }

  for (auto& [graph_idx, tensors] : kv_tensors) m_kvmanager->registerTensors(graph_idx, tensors);

  m_kvmanager->initComplete(m_kv_dim, m_ctx_size, anchor_bitwidth, m_longcontext, m_kv_use_scatter);
  if (m_kvmanager->failed()) {
    State::fatal(m_kvmanager->error());
    return false;
  }

  m_kvmanager->dispatchUpdate(0);
  if (m_kvmanager->failed()) {
    State::fatal(m_kvmanager->error());
    return false;
  }
  return true;
}

inline bool QnnNspModel::updateTensorPointer(GraphVariant& variant,
                                             std::string& key,
                                             QnnUtils::Tensor*& t) {
  QnnUtils::Tensor* tensor_ptr = variant.getInput(key);
  if (tensor_ptr == nullptr) return true;
  if (t == nullptr) t = tensor_ptr;
  if (getBuffer(t) == getBuffer(tensor_ptr)) return true;

  __ERROR("{} has different addresses: {} vs {}", key, (void*)t, (void*)tensor_ptr);
  return false;
}

bool QnnNspModel::initializeTensorPointers() {
  // Ideally this needs to be done for all sets of AR-n available, e.g. for AR-1 and AR-1024

  bool status = true;
  for (auto& variant : m_variant_list) {
    status &= updateTensorPointer(variant, m_layerNames[LayerType::INPUT], t_input_ids);
    status &= updateTensorPointer(variant, m_layerNames[LayerType::ATTN_MASK], t_attn_mask);
    status &= updateTensorPointer(variant, m_layerNames[LayerType::POS_SIN], t_position_ids_sin);
    status &= updateTensorPointer(variant, m_layerNames[LayerType::POS_COS], t_position_ids_cos);
    status &= updateTensorPointer(variant, m_layerNames[LayerType::POS_IDS], t_position_ids);
    status &=
        updateTensorPointer(variant, m_layerNames[LayerType::TOKEN_TYPE_IDS], t_token_type_ids);
    status &= updateTensorPointer(variant, m_layerNames[LayerType::CACHE_INDEX], t_cache_index);
    status &= updateTensorPointer(variant, m_layerNames[LayerType::VALID_MASK], t_valid_mask);

    if (!t_cache_index) {
      std::string cache_position = "cache_position";
      status &= updateTensorPointer(variant, cache_position, t_cache_index);
    }
  }
  if (!status) __ERROR("qnn-htp: Error in setting up named tensor pointers.");

  status &= !(!t_input_ids || !t_attn_mask);
  if (!t_input_ids) __ERROR("Tensor not found: {}", m_layerNames[LayerType::INPUT]);
  if (!t_attn_mask) __ERROR("Tensor not found: {}", m_layerNames[LayerType::ATTN_MASK]);

  if (m_modelArchitectureType ==
      ModelArchitectureType::ENCODER) {  // This input only valid for Encoder only model like bert.
    status &= !(!t_token_type_ids);
    if (!t_token_type_ids) __ERROR("Tensor not found: {}", m_layerNames[LayerType::TOKEN_TYPE_IDS]);
  }

  if (m_positional_encoding.type == PositionalEncoding::ROPE) {
    status &= !(!t_position_ids_sin || !t_position_ids_cos);
    if (!t_position_ids_sin) __ERROR("Tensor not found: {}", m_layerNames[LayerType::POS_SIN]);
    if (!t_position_ids_cos) __ERROR("Tensor not found: {}", m_layerNames[LayerType::POS_COS]);
  } else if (m_positional_encoding.type == PositionalEncoding::ABSOLUTE) {
    status &= !(!t_position_ids);
    if (!t_position_ids) __ERROR("Tensor not found: {}", m_layerNames[LayerType::POS_IDS]);
  } else if (m_positional_encoding.type == PositionalEncoding::ALIBI) {
    status &= !(!t_position_ids);
    if (!t_position_ids) __ERROR("Tensor not found: {}", m_layerNames[LayerType::POS_IDS]);
  } else {
    __ERROR("Unknown Rope Type found for tensor: {}", m_layerNames[LayerType::POS_IDS]);
  }

  // Detect activation bitwidth
  if (status) {
    // Check Input-> Input_ID or Input_Embed
    d_input = t_input_ids->dtype;
    if (!supported_activations.contains(d_input)) {
      __ERROR("Input Tensor: {} as unsupported activation type {}",
              m_layerNames[LayerType::INPUT],
              d_input.str());
      status = false;
    }
    // Check Attention Mask
    d_attn_map = t_attn_mask->dtype;
    if (!supported_activations.contains(d_attn_map)) {
      __ERROR("attention_mask has unsupported type {}", d_attn_map.str());
      status = false;
    }

    int attn_bitwidth   = d_kv.bw();
    bool attn_quantized = d_kv.type() != 2;
    if (attn_quantized) {
      // Support uint8, uint16 and uint32
      if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
        m_attention_positive_value.u32 = 1;  // This sets u8=u16=u32=1
      } else {
        m_attention_positive_value.u32 = 0xffffffff;  // This sets u8=0xff u16=0xffff
      }
      m_attention_negative_value.u32 = 0;  // This sets u8=u16=u32=0
    } else {
      // Support float16 or float32
      m_attention_positive_value.u32 = 0;  // Set u16=u32=0 for fp16 or fp32
      if (attn_bitwidth == 1) {            // float8 is not currently supported
        status = false;
      } else if (attn_bitwidth == 2)
        m_attention_negative_value.u16 = fp16_ieee_from_fp32_value(-1000.0f);
      else if (attn_bitwidth == 4)
        *reinterpret_cast<float*>(&m_attention_negative_value.u32) = -1000.0f;
    }

    // For Encoder only model, Check for Token_type_ids
    if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
      d_token_type = t_token_type_ids->dtype;
      if (!supported_activations.contains(d_token_type)) {
        __ERROR("token_type_ids has unsupported type {}", d_token_type.str());
        status = false;
      }
    }

    // For Position_IDs check data bitWidth
    if (m_positional_encoding.type == PositionalEncoding::ROPE)
      d_pos = t_position_ids_sin->dtype;
    else if (m_positional_encoding.type == PositionalEncoding::ABSOLUTE)
      d_pos = t_position_ids->dtype;
    else if (m_positional_encoding.type == PositionalEncoding::ALIBI)
      d_pos = t_position_ids->dtype;

    if (((m_positional_encoding.type == PositionalEncoding::ABSOLUTE ||
          m_positional_encoding.type == PositionalEncoding::ALIBI) &&
         d_pos != QNN_DATATYPE_INT_32) ||
        (m_positional_encoding.type == PositionalEncoding::ROPE &&
         !supported_activations.contains(d_pos))) {
      __ERROR("position encoding tensor has unsupported type {}", d_pos.str());
      status = false;
    }

    if (t_valid_mask != nullptr && t_valid_mask->dtype != QNN_DATATYPE_UFIXED_POINT_16) {
      __ERROR("Valid mask tensor has unsupported type {}", t_valid_mask->dtype.str());
      status = false;
    }

    __DEBUG("qnn-htp datatypes: d_input {} d_attn_map {} d_pos {} d_kv {}",
            d_input.str(),
            d_attn_map.str(),
            d_pos.str(),
            d_kv.str());

    if (!status) __ERROR("Only 8-bit, 16-bit and 32-bit activations are supported");
  }

  return status;
}

template <typename DType>
void QnnNspModel::setupAttentionMask(const InferenceStep& step,
                                     int32_t start,
                                     const std::vector<int32_t>& attention_map,
                                     int32_t map_n_past,
                                     int32_t map_n_inputs) {
  DType pos_val, neg_val;
  if constexpr (std::is_same_v<DType, uint8_t>) {
    pos_val = m_attention_positive_value.u8;
    neg_val = m_attention_negative_value.u8;
  } else if constexpr (std::is_same_v<DType, uint16_t>) {
    pos_val = m_attention_positive_value.u16;
    neg_val = m_attention_negative_value.u16;
  } else {
    pos_val = m_attention_positive_value.u32;
    neg_val = m_attention_negative_value.u32;
  }

  DType* attn_buffer = (DType*)getBuffer(t_attn_mask);
  // Clear attention mask

  if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
    const size_t n_valid = step.n_valid_kv + step.n_process;
    const size_t offset  = (step.variant == step.ctx_size) ? step.ctx_size - n_valid : 0;
    std::fill_n(attn_buffer, step.ctx_size, neg_val);
    std::fill_n(attn_buffer + offset, n_valid, pos_val);
    return;
  }

  std::fill_n(attn_buffer, step.variant * step.ctx_size, neg_val);

  DType* past_ptr = &attn_buffer[step.past_idx];
  DType* new_ptr  = &attn_buffer[step.new_idx];

  const int32_t map_row_size = map_n_past + map_n_inputs;
  for (int i = 0; i < step.n_process; i++) {
    if (attention_map.empty()) {
      const int32_t n_skip = (start + i < _offset_to_apply_kv_prefix) ? _size_to_skip_kv_prefix : 0;
      std::fill_n(past_ptr + n_skip, step.n_valid_kv - n_skip, pos_val);  // Attend to past KV$
      std::fill_n(new_ptr, i + 1, pos_val);                               // Attend to new tokens
    } else {
      const int32_t* map_row = &attention_map[(start + i) * map_row_size];
      for (int j = 0; j < map_n_past; j++)
        if (map_row[j]) past_ptr[j] = pos_val;
      for (int j = 0; j < map_n_inputs; j++)
        if (map_row[map_n_past + j]) new_ptr[j] = pos_val;
    }
    past_ptr += step.ctx_size;
    new_ptr += step.ctx_size;
  }
}

template <typename DType>
bool QnnNspModel::setupAlibiPositionEmbedding(const InferenceStep& step) {
  DType* alibi_buffer = (DType*)getBuffer(t_position_ids);
  const DType pad_val = static_cast<DType>(step.ctx_size);

  // Clear alibi buffer
  std::fill_n(alibi_buffer, step.variant * step.ctx_size, pad_val);

  // Detect start of past tokens and new tokens based on ctx_size and n_tokens (variant)
  DType* alibi_past = &alibi_buffer[step.past_idx];
  DType* alibi_new  = &alibi_buffer[step.new_idx];

  // Fill alibi positions from [-n_past-i, -i) and [-i, 0]
  for (int i = 0; i < step.n_process; i++) {
    std::iota(std::reverse_iterator<DType*>(alibi_past + step.n_past),
              std::reverse_iterator<DType*>(alibi_past),
              i + 1);  // Fill past tokens
    std::iota(std::reverse_iterator<DType*>(alibi_new + i + 1),
              std::reverse_iterator<DType*>(alibi_new),
              0);  // Fill new tokens

    alibi_past += step.ctx_size;  // Update pointers to next row
    alibi_new += step.ctx_size;
  }

  return true;
}

bool QnnNspModel::setupInputEmbeddings(const InferenceStep& step,
                                       const bool pad_left,
                                       const std::vector<uint8_t>& eagle_embed,
                                       const uint16_t* eagle_feature_in,
                                       const std::vector<int32_t>& selected,
                                       const int32_t start_idx,
                                       const int32_t embed_in_idx,
                                       const bool post_update) {
  size_t in_buf_offset     = 0;
  uint16_t* embed_ptr      = (uint16_t*)getBuffer(t_input_ids);
  uint16_t* eagle_embed_in = (uint16_t*)(eagle_embed.data());
  uint16_t embedDraft      = getEmbeddingBufferSize();
  uint16_t count           = eagle_embed.size() / embedDraft;
  size_t offset_len        = embedDraft;
  size_t feature_len       = embedDraft;
  size_t embed_len         = embedDraft;
  uint16_t increm          = m_embedding_length;
  embed_ptr += start_idx * offset_len;

  if (selected.size() == 0) {
    if (eagle_extra_feature == nullptr) {
      eagle_extra_feature = (uint16_t*)malloc(feature_len);
      // clear the extra feature buffer
      std::memset(eagle_extra_feature, 0, feature_len);
    } else {
      const uint16_t* embed_data = (eagle_embed_in);
      // concat the embedding and feature vector data
      std::memcpy(embed_ptr, embed_data, embed_len);
      std::memcpy(embed_ptr + embed_len / 2, eagle_extra_feature, feature_len);
    }
    embed_ptr += offset_len;
    for (size_t i = 1; i < step.variant; i++) {
      // update the pointer to next token embedding data
      const uint16_t* embed_data = (eagle_embed_in + i * increm);
      // feature data of one before token to copied
      const uint16_t* feature_data = eagle_feature_in + (i - 1 - in_buf_offset) * feature_len / 2;
      // concat the embedding and feature vector data
      std::memcpy(embed_ptr, embed_data, embed_len);
      std::memcpy(embed_ptr + embed_len / sizeof(uint16_t), feature_data, feature_len);
      embed_ptr += offset_len;
    }
    std::memcpy(eagle_extra_feature, eagle_feature_in + step.n_process - 1, feature_len);
  } else {
    if (selected.size() != count && selected.size() != count + 1) {
      __ERROR("setupInputEmbeddings ERROR: wrong selected vector size");
      return false;
    }
    if (eagle_extra_feature == nullptr) {
      eagle_extra_feature = (uint16_t*)malloc(feature_len);
      std::memset(eagle_extra_feature, 0, feature_len);
    }
    const uint16_t* embed_data   = nullptr;
    const uint16_t* feature_data = nullptr;
    size_t copy_buffer_size =
        embed_in_idx + step.variant <= count ? embed_in_idx + step.variant : count;

    for (size_t j = embed_in_idx + start_idx; j < copy_buffer_size; j++) {
      // update the pointer to next token embedding data
      embed_data = (eagle_embed_in + j * increm);
      if (selected[j] >= 0) {
        // Feature data to be copied as per selected idx,each sequence only see the parent and
        // and its predecessor
        feature_data = eagle_feature_in + (selected[j]) * (feature_len / sizeof(uint16_t));
      } else {
        // if selection id is -1 used the last iteration last feature vector.
        feature_data = eagle_extra_feature;
      }
      // concat the embedding and feature vector data
      std::memcpy(embed_ptr, embed_data, embed_len);
      std::memcpy(embed_ptr + embed_len / sizeof(uint16_t), feature_data, feature_len);
      embed_ptr += offset_len;
    }

    if (!post_update) {
      int feature_end_idx =
          copy_buffer_size == embed_in_idx + step.variant ? copy_buffer_size - 1 : copy_buffer_size;
      feature_data =
          eagle_feature_in + (feature_end_idx - embed_in_idx - in_buf_offset) * feature_len / 2;
      // store the extra feature buffer to be used in next iteration, if selection if is -1
      std::memcpy(eagle_extra_feature, feature_data, feature_len);
    }
  }

  return true;
}

bool QnnNspModel::setupInput(const InferenceStep& step,
                             int32_t start,
                             const std::vector<int32_t>& tokens,
                             std::vector<uint8_t>& embeddings,
                             const uint16_t* featureVector,
                             const std::vector<int32_t>& selected,
                             const int32_t start_idx,
                             const bool post_update,
                             const std::vector<int32_t>& attention_map,
                             int32_t map_n_past,
                             int32_t map_n_inputs) {
  const auto& [variant, ctx_size, n_past, n_valid_kv, n_process, past_idx, new_idx] = step;

  if (!tokens.empty()) {
    // Setup input id tensor
    uint32_t* input_id_buffer = (uint32_t*)getBuffer(t_input_ids);
    std::fill_n(input_id_buffer, variant, static_cast<uint32_t>(m_pad_token));
    if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
      const size_t pad_offset = (variant == ctx_size) ? variant - n_process : 0;
      std::memcpy(&input_id_buffer[pad_offset], &tokens[start], n_process * sizeof(uint32_t));
    } else if (step.variant == step.ctx_size) {
      // Special handling for AR-c models. All past tokens must be re-processed
      const size_t n_history = token_history.size();
      std::memcpy(input_id_buffer, token_history.data(), n_history * sizeof(uint32_t));
      std::memcpy(
          &input_id_buffer[n_history], &tokens[start], (n_process - n_history) * sizeof(uint32_t));
    } else {
      // For normal cases (variant < ctx_size), tokens are processed normally
      std::memcpy(input_id_buffer, &tokens[start], n_process * sizeof(uint32_t));
    }
  } else if (!embeddings.empty() &&
             featureVector == nullptr) {  // Quantize and fill, don't make double copy
    if (embedding_datatype == "QNN_DATATYPE_FLOAT_32") {
      // First flush the buffer with eos token embedding
      float* embeddingSrc = (float*)m_eosEmbedding.data();
      for (size_t i = n_process; i < variant; i++) {
        quantizeInput(embeddingSrc, i * m_embd_size, m_embd_size);
      }

      // Quantize the data input vector
      embeddingSrc = (float*)embeddings.data();
      quantizeInput(&embeddingSrc[start * m_embd_size], 0, n_process * m_embd_size);
    } else {
      // Size of the buffer for one embedding vector.
      const size_t embedBufSize = m_embeddingBufferSize;
      // First flush the buffer with eos token embedding
      uint8_t* embeddingSrc = static_cast<uint8_t*>(m_eosEmbedding.data());
      for (size_t i = n_process; i < variant; i++) {
        std::copy(embeddingSrc,
                  embeddingSrc + embedBufSize,
                  (uint8_t*)getBuffer(t_input_ids) + i * embedBufSize);
      }

      // Copy the data input vector
      embeddingSrc = static_cast<uint8_t*>(embeddings.data()) + (start * embedBufSize);
      std::copy(embeddingSrc,
                embeddingSrc + (n_process * embedBufSize),
                (uint8_t*)getBuffer(t_input_ids));
    }
  } else if (!embeddings.empty() && featureVector != nullptr) {
    setupInputEmbeddings(
        step, false, embeddings, featureVector, selected, start_idx, start, post_update);
  }

  // Set up the input scatter index as new_idx (i.e. the index where new KV$ is stored)
  if (t_cache_index != nullptr) {
    uint32_t* cache_index_buffer = (uint32_t*)getBuffer(t_cache_index);
    std::iota(cache_index_buffer,
              cache_index_buffer + t_cache_index->dims.getNumElements(),
              step.new_idx);
  } else {
    __DEBUG("cache position not detected");
  }

  if (t_valid_mask != nullptr) {  // Set up a valid mask. Assumes u16 datatype (from validateModel)
    // Quantize mask value to u16
    const auto [scale, offset] = t_valid_mask->quantParam[0];
    uint16_t mask_val = QnnUtils::quantize<double, uint16_t>(1.0 / n_process, offset, scale);

    bool hasSpeculativeTokens = false;
    if (!tokens.empty()) {
      for (int32_t token : tokens) {
        if (token >= m_vocab_size) {
          hasSpeculativeTokens = true;
          break;
        }
      }
    }

    const size_t n_masked = hasSpeculativeTokens ? 1 : n_process;
    // Setup the buffer
    uint16_t* mask_buffer = (uint16_t*)getBuffer(t_valid_mask);
    std::fill_n(mask_buffer, n_masked, mask_val);
    std::memset(&mask_buffer[n_masked], 0, (variant - n_masked) * sizeof(uint16_t));
  }

  // Setup the attention mask correctly
  if (d_attn_map.bw() == 1)
    setupAttentionMask<uint8_t>(step, start, attention_map, map_n_past, map_n_inputs);
  else if (d_attn_map.bw() == 2)
    setupAttentionMask<uint16_t>(step, start, attention_map, map_n_past, map_n_inputs);
  else if (d_attn_map.bw() == 4)
    setupAttentionMask<uint32_t>(step, start, attention_map, map_n_past, map_n_inputs);

  // Setup token type IDs
  if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
    // BERT Specific
    uint32_t* token_type_id_buffer = (uint32_t*)getBuffer(t_token_type_ids);
    std::memset(token_type_id_buffer, 0, step.variant * sizeof(uint32_t));
  }

  if (m_positional_encoding.type == PositionalEncoding::ROPE) {
    // Simple RoPE position ID setup
    std::vector<int32_t> position_ids(variant, 0);
    if (attention_map.empty()) {
      std::iota(&position_ids[0], &position_ids[n_process], n_past - _size_to_skip_kv_prefix);
    } else {
      // Position IDs need to accumulate the virtual token count.
      const int32_t virtualPastOffset = step.n_past - map_n_past;
      const int32_t map_row_size      = map_n_past + map_n_inputs;
      for (int i = 0; i < n_process; i++) {
        const int32_t* map_row = &attention_map[(start + i) * map_row_size];
        position_ids[i]        = std::accumulate(  // PositionID = #tokens attended
                              &map_row[_size_to_skip_kv_prefix],  // Skip attention for forecast KV$
                              &map_row[map_n_past + n_process],  //
                              -map_row[map_n_past + i]  // PositionID skips attending itself
                              ) +
                          virtualPastOffset;
      }
    }
    // TODO: Compile position_ids to translate from [0,1,2,3,4,0,0,...,0] -> [(0,5), (0,1),...]
    // This is to batch the memory copy calls together, which is more optimal (theoretically)
    int rope_bitwidth      = d_pos.bw();
    uint8_t* cos_buffer    = (uint8_t*)getBuffer(t_position_ids_cos);
    uint8_t* sin_buffer    = (uint8_t*)getBuffer(t_position_ids_sin);
    const size_t rope_size = m_pos_dim * rope_bitwidth;
    for (int i = 0; i < variant; i++) {
      const size_t src_offset = position_ids[i] * rope_size;
      const size_t dst_offset = i * rope_size;
      std::memcpy(&sin_buffer[dst_offset], (uint8_t*)rope_sin + src_offset, rope_size);
      std::memcpy(&cos_buffer[dst_offset], (uint8_t*)rope_cos + src_offset, rope_size);
    }
  } else if (m_positional_encoding.type == PositionalEncoding::ABSOLUTE) {
    uint32_t* position_id_buffer = (uint32_t*)getBuffer(t_position_ids);
    std::memset(position_id_buffer, 0, step.variant * sizeof(uint32_t));

    // Fill up position_ids buffer
    const size_t pad_offset =
        (m_modelArchitectureType == ModelArchitectureType::ENCODER) ? variant - n_process : 0;

    uint32_t* pos_id_start = &position_id_buffer[pad_offset];
    uint32_t* pos_id_end   = pos_id_start + step.n_process;
    std::iota(pos_id_start, pos_id_end, step.n_past);
  } else if (m_positional_encoding.type == PositionalEncoding::ALIBI) {
    setupAlibiPositionEmbedding<int32_t>(step);
  }
  return true;
}

// Parses a 1D attention mask into a 2D attention mask
std::vector<int32_t> QnnNspModel::parseAttentionMask(int32_t n_inputs,
                                                     int32_t n_past,
                                                     const std::vector<int32_t>& attention_map) {
  if (attention_map.size() != n_inputs) return {};

  const int32_t row_size = n_past + n_inputs;  // Each input attends to all past+new inputs
  std::vector<int32_t> new_map(n_inputs * row_size, 0);

  for (int i = 0; i < n_inputs; i++) {
    auto cur_row = &new_map[i * row_size];
    if (attention_map[i] < 0)  // Negative value means attend to the last n-th tokens only
      std::fill_n(cur_row, n_past + attention_map[i] + 1, 1);
    else  // Positive value means copy the attention mask from a previous row
      std::memcpy(cur_row, &new_map[attention_map[i] * row_size], row_size * sizeof(int32_t));
    cur_row[n_past + i] = 1;  // All tokens attend to itself
  }

  // For SSD, first _offset_to_apply_kv_prefix inputs skip the first _size_to_skip_kv_prefix KV$
  for (int i = 0; i < std::min(n_inputs, _offset_to_apply_kv_prefix); i++)
    std::fill_n(&new_map[i * row_size], _size_to_skip_kv_prefix, 0);

  return new_map;
}

inline void QnnNspModel::syncDrafTargetPrefill(bool isDraft, bool isReset) {
  if (_counter == nullptr) {
    return;
  }
  if (isReset == false) {
    if (isDraft == true) {
      while (_counter != nullptr && *_counter == 0) {
      }
    } else {
      while (_counter != nullptr && *_counter != 0) {
      }
    }
  } else {
    if (isDraft) {
      if (_counter != nullptr) {
        *_counter = 0;
      }
    } else {
      if (_counter != nullptr) {
        *_counter = 1;
      }
    }
  }
}

size_t QnnNspModel::runInference(const std::vector<int32_t>& tokens,
                                 std::vector<uint8_t>& embedding,
                                 const uint16_t* featureVector,
                                 const std::vector<int32_t>& selected,
                                 const int32_t start_idx,
                                 const bool post_update,
                                 const std::vector<int32_t>& attention_map,
                                 std::vector<float>& output,
                                 bool output_all) {
  qualla::Timer start;
  __TRACE("runInference logits_all={} tokens={} featureVector {}",
          output_all,
          tokens,
          (uint64_t)featureVector);

  bool draft = false;
  if (featureVector != nullptr) {
    draft = true;
  }

  if ((tokens.size() == 0) && (embedding.size() == 0)) return 0;

  size_t embedBufSize    = m_embeddingBufferSize;
  int32_t embeddingCount = embedding.size() / embedBufSize;

  // Disable token_history (required for AR-c models) if embedding input is processed
  if (embeddingCount > 0) token_history_enabled = false;

  // Create a strategy to run the inference
  const int32_t n_inputs   = static_cast<int32_t>(tokens.size() + embeddingCount);
  const int32_t map_n_past = m_kvmanager->n_valid_kv();
  if (!m_kvmanager->prepareInferenceStrategy(n_inputs)) {
    State::fatal(m_kvmanager->error());
    return false;
  }

  // Parse a 1D attention map into a 2D attention map
  auto attention_2d = parseAttentionMask(n_inputs, map_n_past, attention_map);
  auto& parsed_map  = (attention_map.size() == n_inputs) ? attention_2d : attention_map;

  // user choice overwrites the default behaviour in case of Embedding models
  if (m_modelArchitectureType == ModelArchitectureType::ENCODER) output_all = !m_pooled_output;

  size_t output_size = output_all ? n_inputs : 1;  // actual number of logits

  if (m_modelArchitectureType == ModelArchitectureType::ENCODER)
    output.resize(output_size * m_embd_size);
  else
    output.resize(output_size * m_vocab_size);

  __TRACE("runInference output ={}", output.size());

  // Loop over each planned iteration in the inference strategy
  InferenceStep step;
  int32_t n_processed = 0;  // Number of total tokens processed so far

  while (m_kvmanager->nextInferenceStep(step)) {
    __DEBUG("Inference step: {}", step.str());
    syncDrafTargetPrefill(draft, false);
    if (!setupInput(step,
                    n_processed,
                    tokens,
                    embedding,
                    featureVector,
                    selected,
                    start_idx,
                    post_update,
                    parsed_map,
                    map_n_past,
                    n_inputs))
      return false;

    uint8_t indx = 0;
    for (auto& nsp_graph : m_nsp_graphs) {
      const int graph_idx = nsp_graph.idx();

      if (nsp_graph.m_graphType == GraphType::LMHEAD && (!output_all) &&
          !m_kvmanager->isFinalInferenceStep()) {  // Must meet all the criteria to skip LMHEAD
        continue;
      }

      if (!m_kvmanager->block(Scope::per_graph(graph_idx))) {
        State::error(m_kvmanager->error());
        return false;
      }

      if (!nsp_graph.execute(
              step.variant, step.ctx_size, m_inference_count, graph_switching, lazy_lora)) {
        fatal(fmt::format("Failed to execute graph {}", graph_idx));
        return false;
      }

      if (!m_kvmanager->unblock(Scope::per_graph(graph_idx))) {
        State::error(m_kvmanager->error());
        return false;
      }
    }
    if (m_modelArchitectureType != ModelArchitectureType::ENCODER && output_all)
      getDequantLogits(
          std::span(&output[n_processed * m_vocab_size], step.n_process * m_vocab_size),
          step,
          step.n_process);

    // Debug dump outputs
    if (_debug_outputs) {
      if (m_modelArchitectureType == ModelArchitectureType::ENCODER) {
        debugOutputs(step, m_layerNames[LayerType::POOL_OUTPUT]);
        debugOutputs(step, m_layerNames[LayerType::SEQ_OUTPUT]);
      } else {
        debugOutputs(step, m_layerNames[LayerType::OUTPUT]);
      }
    }

    n_processed += step.n_process;
    m_inference_count++;
    syncDrafTargetPrefill(draft, true);
  }

  if (post_update) {
    updateFeatureBuffer(embeddingCount);
  }

  // If only the last output is required, then process this request here
  if (m_modelArchitectureType != ModelArchitectureType::ENCODER) {
    if (!output_all) {
      getDequantLogits(std::span{output.data(), output.size()}, step, 1);
    }
  } else {
    getEmbeddings(std::span{output.data(), output.size()}, step);
  }

  // Maintain a history of all processed tokens
  if (token_history_enabled)
    token_history.insert(token_history.end(), tokens.begin(), tokens.end());

  __DEBUG("qnn-htp: run-inference complete : {} usec ", start.elapsed_usec());
  return output_size;
}

size_t QnnNspModel::runInference(const std::vector<int32_t>& tokens,
                                 std::vector<uint8_t>& embedding,
                                 const uint16_t* featureVector,
                                 const std::vector<int32_t>& selected,
                                 const int32_t start_idx,
                                 const bool post_update,
                                 const std::vector<int32_t>& attention_map,
                                 Tensor& output,
                                 bool output_all) {
  qualla::Timer start;

  __TRACE("runInference logits_all={} tokens={}", output_all, tokens);
  if ((tokens.size() == 0) && (embedding.size() == 0)) return 0;

  const size_t embedBufSize = m_embeddingBufferSize;
  int32_t embeddingCount    = embedding.size() / embedBufSize;
  // Disable token_history (required for AR-c models) if embedding input is processed
  if (embeddingCount > 0) token_history_enabled = false;

  bool draft = false;
  if (featureVector != nullptr) {
    draft = true;
  }

  // Create a strategy to run the inference
  const int32_t n_inputs = static_cast<int32_t>(tokens.size() + embeddingCount);
  const int32_t n_past   = m_kvmanager->n_past();
  if (!m_kvmanager->prepareInferenceStrategy(n_inputs)) {
    State::fatal(m_kvmanager->error());
    return false;
  }

  // Parse a 1D attention map into a 2D attention map
  auto attention_2d = parseAttentionMask(n_inputs, n_past, attention_map);
  auto& parsed_map  = (attention_map.size() == n_inputs) ? attention_2d : attention_map;

  size_t output_size = output_all ? n_inputs : 1;  // actual number of logits

  output.setSize(0);

  // Loop over each planned iteration in the inference strategy
  InferenceStep step;
  int32_t n_processed = 0;  // Number of total tokens processed so far

  bool requireLogitsCopy = false;
  if (m_kvmanager->getStrategySize() > 1 && output_all && draft == false) {
    requireLogitsCopy = true;
  }
  requireLogitsCopy = false;
  while (m_kvmanager->nextInferenceStep(step)) {
    __DEBUG("Inference step: {}", step.str());
    syncDrafTargetPrefill(draft, false);
    if (!setupInput(step,
                    n_processed,
                    tokens,
                    embedding,
                    featureVector,
                    selected,
                    start_idx,
                    post_update,
                    parsed_map,
                    n_past,
                    n_inputs))
      return false;

    for (auto& nsp_graph : m_nsp_graphs) {
      const int graph_idx = nsp_graph.idx();

      if (!m_kvmanager->block(Scope::per_graph(graph_idx))) {
        State::error(m_kvmanager->error());
        return false;
      }

      if (!nsp_graph.execute(
              step.variant, step.ctx_size, m_inference_count, graph_switching, lazy_lora)) {
        fatal(fmt::format("Failed to execute graph {}", graph_idx));
        return false;
      }

      if (!m_kvmanager->unblock(Scope::per_graph(graph_idx))) {
        State::error(m_kvmanager->error());
        return false;
      }
    }

    if (output_all) getLogits(output, step, step.n_process, requireLogitsCopy);

    // Debug dump outputs
    if (_debug_outputs) {
      debugOutputs(step, m_layerNames[LayerType::OUTPUT]);
    }

    n_processed += step.n_process;
    m_inference_count++;
    syncDrafTargetPrefill(draft, true);
  }
  if (post_update) {
    updateFeatureBuffer(embeddingCount);
  }
  // If only the last output is required, then process this request here
  if (!output_all) {
    getLogits(output, step, 1);
  }

  // Maintain a history of all processed tokens
  if (token_history_enabled)
    token_history.insert(token_history.end(), tokens.begin(), tokens.end());

  __DEBUG("qnn-htp: run-inference complete : {} usec ", start.elapsed_usec());
  return output_size;
}

void QnnNspModel::updateFeatureBuffer(int32_t embeddingCount) {
  uint32_t feature_len = m_embedding_length * sizeof(uint16_t);
  if (eagle_extra_feature == nullptr) {
    eagle_extra_feature = (uint16_t*)malloc(feature_len);
    std::memset(eagle_extra_feature, 0, feature_len);
  }
  void* eagle_feature = nullptr;
  getIOBufferByName(draftFeatureName, eagle_feature, false);
  const uint16_t* feature_data = nullptr;
  int feature_offset           = embeddingCount - 1;

  feature_data = (uint16_t*)eagle_feature + feature_offset * feature_len / sizeof(uint16_t);
  std::memcpy(eagle_extra_feature, feature_data, feature_len);
}

// Dumps out the specified tensor to _debug_path numbered according to m_inference_count
bool QnnNspModel::debugOutputs(const InferenceStep& step, const std::string& tensor_name) {
  GraphVariant* graph_variant = m_nsp_graphs.back()(step.variant, step.ctx_size);
  QnnUtils::Tensor* tensor    = graph_variant->getOutput(tensor_name);
  if (tensor == nullptr) {
    __DEBUG("qnn-htp: Couldn't find tensor {} in graph {}", tensor_name, graph_variant->graph_name);
    return false;
  }

  // For ENCODER models, dump the complete buffer. For LLM models, dump the generated logits
  const int output_bitwidth = tensor->dtype.bw();  // Detect 8-bit vs 16-bit logits
  const int32_t output_size = (m_modelArchitectureType == ModelArchitectureType::ENCODER)
                                  ? step.ctx_size * m_embd_size * output_bitwidth
                                  : step.n_process * m_vocab_size * output_bitwidth;

  std::string fname = fmt::format("{}/{}/{:03d}", _debug_path, tensor_name, m_inference_count);
  if (!QnnUtils::writeRawData(getBuffer(tensor), output_size, fname)) {
    __DEBUG("qnn-htp: Failed to save {}. Error when writing to {}", tensor_name, fname);
    return false;
  }
  return true;
}

bool QnnNspModel::quantizeInput(float* in, size_t tensorOffset, size_t length) {
  if (t_input_ids == nullptr) {
    __ERROR("Input Tensor {} not found during execute", m_layerNames[LayerType::INPUT]);
    return false;
  }

  const auto scale  = t_input_ids->quantParam[0].scale;
  const auto offset = t_input_ids->quantParam[0].offset;

  switch (t_input_ids->dtype) {
    case QNN_DATATYPE_UFIXED_POINT_8:
      QnnUtils::quantizeTensorPtr(
          in, (uint8_t*)getBuffer(t_input_ids) + tensorOffset, offset, scale, length);
      break;
    case QNN_DATATYPE_UFIXED_POINT_16:
      QnnUtils::quantizeTensorPtr(
          in, (uint16_t*)getBuffer(t_input_ids) + tensorOffset, offset, scale, length);
      break;
    default:
      __ERROR("Unsupported alpha tensor dtype {}", t_input_ids->dtype.str());
      return false;
  }

  return true;
}

size_t QnnNspModel::getEmbeddingBufferSize() { return m_embeddingBufferSize; }

void QnnNspModel::getTensorParam(
    LayerType layerType, std::string& dataType, double& scale, int32_t& offset, size_t& bitWidth) {
  if (layerType == LayerType::INPUT) {
    dataType = t_input_ids->dtype.str();
    scale    = t_input_ids->quantParam[0].scale;
    offset   = t_input_ids->quantParam[0].offset;
    bitWidth = t_input_ids->dtype.bw();
  }
}

bool QnnNspModel::cacheEosEmbedding(std::vector<uint8_t>& eosEmbedding) {
  m_eosEmbedding = eosEmbedding;
  return true;
}

bool QnnNspModel::setKVCacheNPast(size_t n_past, const std::vector<bool>& selected) {
  if (!m_kvmanager->dispatchUpdate(n_past, selected)) {
    __ERROR("qnn-htp: KV$ update failed. {}", m_kvmanager->error());
    State::error(m_kvmanager->error());
    return false;
  }

  // Manage the token history based on the KV$ accepted by the user
  // If no selection mask is passed, we can simply resize the history to n_past
  // If a selection mask is passed, we must selectively filter out rejected KV$
  if (token_history_enabled) {  // Token history must be disabled on embedding input or longcontext
    if (selected.empty())
      token_history.resize(n_past);
    else {
      auto it = token_history.begin() + token_history.size() - selected.size();
      for (const bool& isSelected : selected) {
        it = (isSelected) ? it + 1 : token_history.erase(it);  // Erase if not selected, else no-op
      }
    }
  }
  return true;
}

size_t QnnNspModel::getDequantLogits(std::span<float> buffer, InferenceStep& step, int32_t count) {
  qualla::Timer start;

  QnnUtils::Tensor* const spec =
      m_nsp_graphs.back()(step.variant, step.ctx_size)->getOutput(m_layerNames[LayerType::OUTPUT]);

  auto [scale, offset] = spec->quantParam[0];               // Quantization parameters
  auto dtype           = QnnUtils::DataType(spec->tensor);  // Datatype of the generated output
  int bitwidth         = spec->dtype.bw();                  // Number of bytes per output element
  auto logit_buffer    = (uint8_t*)getBuffer(spec);         // Pointer to the actual output data

  if (spec->dims.getNumElements() == m_vocab_size && count > 1) {
    State::error("Requested all logits, but graph only produces one logit");
    return 0;
  }

  // Offset to the appropriate location in the output buffer. Note this assumes right-padded input
  logit_buffer += (step.n_process - count) * m_vocab_size * bitwidth;

  const int size = static_cast<int>(m_vocab_size * count);
  __TRACE("qnn-htp: getDequantLogits Returning {}*{} from [{}]", count, m_vocab_size, step.str());

  switch (dtype) {
    case QNN_DATATYPE_UFIXED_POINT_8:
      deQuantizeOutputs((uint8_t*)logit_buffer, buffer, scale, offset, size);
      break;
    case QNN_DATATYPE_UFIXED_POINT_16:
      deQuantizeOutputs((uint16_t*)logit_buffer, buffer, scale, offset, size);
      break;
    case QNN_DATATYPE_FLOAT_16:
      castOutputs((uint16_t*)logit_buffer, buffer, size, bitwidth);
      break;
    case QNN_DATATYPE_FLOAT_32:
      castOutputs((float*)logit_buffer, buffer, size, bitwidth);
      break;
    default:
      State::error(fmt::format("Unsupported logits dtype {}", dtype.str()));
      return 0;
  }

  __DEBUG("qnn-htp: getDequantLogits complete. Returning {} outputs in {} usec",
          count,
          start.elapsed_usec());
  return size;
}

size_t QnnNspModel::getLogits(Tensor& logits,
                              InferenceStep& step,
                              int32_t count,
                              bool requireLogitsCopy) {
  qualla::Timer start;

  QnnUtils::Tensor* const spec =
      m_nsp_graphs.back()(step.variant, step.ctx_size)->getOutput(m_layerNames[LayerType::OUTPUT]);

  auto [scale, offset] = spec->quantParam[0];               // Quantization parameters
  auto dtype           = QnnUtils::DataType(spec->tensor);  // Datatype of the generated output
  int bitwidth         = spec->dtype.bw();                  // Number of bytes per output element
  auto logit_buffer    = (uint8_t*)getBuffer(spec);         // Pointer to the actual output data

  if (spec->dims.getNumElements() == m_vocab_size && count > 1) {
    State::error("Requested all logits, but graph only produces one logit");
    return 0;
  }

  // Offset to the appropriate location in the output buffer. Note this assumes right-padded input
  logit_buffer += (step.n_process - count) * m_vocab_size * bitwidth;

  const int size = static_cast<int>(m_vocab_size * count);
  __TRACE("qnn-htp: getLogits Returning {}*{} from [{}]", count, m_vocab_size, step.str());

  switch (dtype) {
    case QNN_DATATYPE_UFIXED_POINT_8: {
      if (requireLogitsCopy) {
        logits.logits.reserve(logits.getSize() + size);
        uint8_t* logit_buffer_u8 = (uint8_t*)logit_buffer;
        for (int i = 0; i < size; i++) {
          logits.logits[logits.getSize() + i] = ((float)logit_buffer_u8[i] + offset) * scale;
        }
        logits.setQuantizationParams(1, 0);
        logits.setData((void*)(logits.logits.data()));
        logits.setSize(logits.getSize() + size);
        logits.setDataType(TENSOR_DATATYPE_FLOAT_32);
      } else {
        logits.setQuantizationParams(scale, offset);
        logits.setData((void*)logit_buffer);
        logits.setSize(size);
        logits.setDataType(TENSOR_DATATYPE_UFIXED_POINT_8);
      }
      break;
    }
    case QNN_DATATYPE_UFIXED_POINT_16: {
      if (requireLogitsCopy) {
        logits.logits.reserve(logits.getSize() + size);
        uint16_t* logit_buffer_u16 = (uint16_t*)logit_buffer;
        for (int i = 0; i < size; i++) {
          logits.logits[logits.getSize() + i] = (((float)logit_buffer_u16[i] + offset) * scale);
        }
        logits.setQuantizationParams(1, 0);
        logits.setData((void*)(logits.logits.data()));
        logits.setSize(logits.getSize() + size);
        logits.setDataType(TENSOR_DATATYPE_FLOAT_32);
      } else {
        logits.setQuantizationParams(scale, offset);
        logits.setData((void*)logit_buffer);
        logits.setSize(size);
        logits.setDataType(TENSOR_DATATYPE_UFIXED_POINT_16);
      }
      break;
    }
    case QNN_DATATYPE_FLOAT_16: {
      if (requireLogitsCopy) {
        logits.logits.reserve(logits.getSize() + size);
        uint16_t* logit_buffer_fp16 = (uint16_t*)logit_buffer;
        for (int i = 0; i < size; i++) {
          logits.logits[logits.getSize() + i] = fp16_ieee_to_fp32_value(logit_buffer_fp16[i]);
        }
        logits.setQuantizationParams(1, 0);
        logits.setData((void*)(logits.logits.data()));
        logits.setSize(logits.getSize() + size);
        logits.setDataType(TENSOR_DATATYPE_FLOAT_32);
      } else {
        logits.setQuantizationParams(1, 0);
        logits.setData((void*)logit_buffer);
        logits.setSize(size);
        logits.setDataType(TENSOR_DATATYPE_FLOAT_POINT_16);
      }
      break;
    }
    case QNN_DATATYPE_FLOAT_32: {
      if (requireLogitsCopy) {
        logits.logits.reserve(logits.getSize() + size);
        float* logit_buffer_fp32 = (float*)logit_buffer;
        for (int i = 0; i < size; i++) {
          logits.logits[logits.getSize() + i] = logit_buffer_fp32[i];
        }
        logits.setData((void*)(logits.logits.data()));
        logits.setSize(logits.getSize() + size);
      } else {
        logits.setData((void*)logit_buffer);
        logits.setSize(size);
      }
      logits.setQuantizationParams(1, 0);
      logits.setDataType(TENSOR_DATATYPE_FLOAT_32);
      break;
    }
    default: {
      State::error(fmt::format("Unsupported logits dtype {}", dtype.str()));
      return 0;
    }
  }

  __DEBUG(
      "qnn-htp: getLogits complete. Returning {} outputs in {} usec", count, start.elapsed_usec());
  return size;
}

bool QnnNspModel::calculate_rope_embeddings(void) {
  if (m_positional_encoding.type != PositionalEncoding::ROPE) return true;

  const size_t nmemb = m_ctx_size * m_pos_dim;
  const int pos_bw   = d_pos.bw();

  const double theta                    = m_positional_encoding.rope_params.theta;
  const RopeScalingParams& rope_scaling = m_positional_encoding.rope_params.rope_scaling;

  rope_sin = malloc(nmemb * pos_bw);
  rope_cos = malloc(nmemb * pos_bw);

  auto [q_scale, q_offset] = t_position_ids_cos->quantParam[0];
  if (d_pos == QNN_DATATYPE_FLOAT_16 ||
      d_pos == QNN_DATATYPE_FLOAT_32) {  // If floating point, don't quantize!
    q_scale  = 1.0;
    q_offset = 0;
  }

  // Calculate inv_freq array
  std::vector<double> inv_freq(m_pos_dim);
  const double exponent = 1.0 / static_cast<double>(m_pos_dim);
  for (int j = 0; j < m_pos_dim; j++) inv_freq[j] = 1.0 / pow(theta, j * exponent);
  double attention_factor = 1.0;
  if (rope_scaling.rope_type == RopeScalingParams::ROPE_LLAMA3) {
    // Implemented from HuggingFace
    // https://github.com/huggingface/transformers/blob/47c29ccfaf56947d845971a439cbe75a764b63d7/src/transformers/modeling_rope_utils.py#L298
    const double& factor           = rope_scaling.llama3_params.factor;
    const double& low_freq_factor  = rope_scaling.llama3_params.low_freq_factor;
    const double& high_freq_factor = rope_scaling.llama3_params.high_freq_factor;
    const int& old_context_len     = rope_scaling.llama3_params.original_max_position_embeddings;

    const double low_freq_wavelen  = old_context_len / low_freq_factor;
    const double high_freq_wavelen = old_context_len / high_freq_factor;

    for (int j = 0; j < m_pos_dim; j++) {
      const double wavelen = 2 * M_PI / inv_freq[j];
      if (wavelen < high_freq_wavelen)  // wavelen < high_freq_wavelen: do nothing
        continue;
      else if (wavelen > low_freq_wavelen)  // wavelen > low_freq_wavelen: divide by factor
        inv_freq[j] = 1.0 / static_cast<double>(factor * pow(theta, j * exponent));
      else {  // otherwise: interpolate between the two, using a smooth factor
        assert(low_freq_wavelen != high_freq_wavelen);
        const double smooth = (static_cast<double>(old_context_len) / wavelen - low_freq_factor) /
                              (high_freq_factor - low_freq_factor);
        inv_freq[j] = ((1 - smooth) * inv_freq[j] / factor + smooth * inv_freq[j]);
      }
    }
  } else if (rope_scaling.rope_type == RopeScalingParams::ROPE_LONGROPE) {
    // Validate factor >= 1.0, len(long_factor) == rope-dim and len(short_factor) == rope-dim
    const double& factor       = rope_scaling.longrope_params.factor;
    const int& old_context_len = rope_scaling.longrope_params.original_max_position_embeddings;

    const auto& inv_factors = (m_ctx_size > old_context_len)
                                  ? rope_scaling.longrope_params.long_factor
                                  : rope_scaling.longrope_params.short_factor;

    if (inv_factors.size() != m_pos_dim)
      throw std::runtime_error(
          fmt::format("long-factor (len={}) and short-factor (len={}) must have length rope-dim={}",
                      rope_scaling.longrope_params.long_factor.size(),
                      rope_scaling.longrope_params.short_factor.size(),
                      m_pos_dim));

    for (int j = 0; j < m_pos_dim; j++) inv_freq[j] = inv_freq[j] / inv_factors[j];

    attention_factor =
        std::sqrt(1.0 + std::log(factor) / std::log(static_cast<double>(old_context_len)));
  }
  for (int i = 0; i < m_ctx_size; i++) {
    for (int j = 0; j < m_pos_dim; j++) {
      const double freq = i * inv_freq[j];

      const double sin_val = ((sin(freq) * attention_factor) / q_scale) - q_offset;
      const double cos_val = ((cos(freq) * attention_factor) / q_scale) - q_offset;

      // round() instead of floor() seems to produce an acuracy drop. To debug later
      switch (d_pos) {
        case QNN_DATATYPE_UFIXED_POINT_8:
          ((uint8_t*)rope_sin)[i * m_pos_dim + j] = static_cast<uint8_t>(sin_val);
          ((uint8_t*)rope_cos)[i * m_pos_dim + j] = static_cast<uint8_t>(cos_val);
          break;
        case QNN_DATATYPE_UFIXED_POINT_16:
          ((uint16_t*)rope_sin)[i * m_pos_dim + j] = static_cast<uint16_t>(sin_val);
          ((uint16_t*)rope_cos)[i * m_pos_dim + j] = static_cast<uint16_t>(cos_val);
          break;
        case QNN_DATATYPE_FLOAT_16:
          ((uint16_t*)rope_sin)[i * m_pos_dim + j] = fp16_ieee_from_fp32_value(sin_val);
          ((uint16_t*)rope_cos)[i * m_pos_dim + j] = fp16_ieee_from_fp32_value(cos_val);
          break;
        case QNN_DATATYPE_FLOAT_32:
          ((float*)rope_sin)[i * m_pos_dim + j] = static_cast<float>(sin_val);
          ((float*)rope_cos)[i * m_pos_dim + j] = static_cast<float>(cos_val);
          break;
        default:
          __ERROR("Unsupported position ids datatype {}", d_pos.str());
          return false;
      }
    }
  }

  if (_debug_tensors) {
    std::string dtype =
        fmt::format("{}{}", (d_pos == QNN_DATATYPE_FLOAT_16) ? "f" : "u", pos_bw * 8);
    std::string fname_sin = fmt::format("{}/position_ids_sin.{}.dat", _debug_path, dtype);
    std::string fname_cos = fmt::format("{}/position_ids_cos.{}.dat", _debug_path, dtype);
    QnnUtils::writeRawData(rope_sin, nmemb * pos_bw, fname_sin);
    QnnUtils::writeRawData(rope_cos, nmemb * pos_bw, fname_cos);
  }

  return true;
}

bool QnnNspModel::load_lmhead_weight_as_input(void) {
  if (!_lmhead_weight_input) return true;
  if (_lmhead_weight_input && lmhead_weight_dir.empty()) {
    __ERROR("NSPModel: LMhead weight file not found");
    return false;
  }
  for (auto& variant : m_variant_list) {
    for (auto& [tname, tspec] : variant.input_specs) {
      if (tname.compare("weight") == 0) {
        // weight tensor file name should be in same format as tensor name present in graph
        std::string weight_file =
            (model_basedir / fs::path(lmhead_weight_dir) / fs::path(tname + ".raw")).string();

        QnnUtils::Dims dims = tspec.dims;
        size_t numElements  = dims.getNumElements();

        size_t size = sizeof(float);
        std::vector<float> weight_f32;  // Temporary variable to load fp32 values
        weight_f32.reserve(numElements);

        FILE* fp = fopen(weight_file.c_str(), "r");
        if (fp == NULL) {
          __ERROR("NSPModel: Error opening file: {}", weight_file);
          return false;
        }

        size_t count = fread(weight_f32.data(), size, numElements, fp);
        fclose(fp);

        if (count != numElements) {
          __ERROR("NSPModel: Could not load {} - expected file size {}",
                  weight_file,
                  numElements * size);
          return false;
        }

        int8_t* weight_buffer = (int8_t*)getBuffer(tspec);
        // Quantize the values, per width quantization
        QnnUtils::perWidthQuantizeTensorPtr(weight_f32.data(),
                                            weight_buffer,
                                            tspec.quantParam,
                                            dims.height,
                                            dims.width,
                                            dims.channel);
      }
    }
  }
  return true;
}

void QnnNspModel::getInputQuantParam(double& scale, int& offset) {
  auto tmp = t_input_ids->quantParam[0];
  scale    = tmp.scale;
  offset   = tmp.offset;
}

size_t QnnNspModel::loadKVCache(const std::string& load_path, bool chooseHigherVariant) {
  m_kvmanager->block(Scope::global());
  size_t ret = m_kvmanager->loadKVCache(load_path);
  if (m_kvmanager->failed()) State::error(m_kvmanager->error());
  return ret;
}

bool QnnNspModel::saveKVCache(const std::string& save_path) {
  m_kvmanager->block(Scope::global());
  bool ret = m_kvmanager->dumpKVCache(save_path);
  if (m_kvmanager->failed()) State::error(m_kvmanager->error());
  return ret;
}

bool QnnNspModel::saveKVCacheToBuffer(Buffer* kvBuff) {
  m_kvmanager->block(Scope::global());
  bool ret = m_kvmanager->dumpKVCache(kvBuff);
  if (m_kvmanager->failed()) State::error(m_kvmanager->error());
  return ret;
}

bool QnnNspModel::getCacheSpec(CacheFileSpec& spec) {
  m_kvmanager->block(Scope::global());
  bool ret = m_kvmanager->getCacheSpec(spec);
  return ret;
}

bool QnnNspModel::getKVHead(
    CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) {
  m_kvmanager->block(Scope::global());
  bool ret = m_kvmanager->getKVHead(spec, layer, head, data, scale);
  return ret;
}

void QnnNspModel::setHigherVariant() {
  auto& [new_variant, _] = nsp_graph_count.rbegin()->first;  // Guarantees largest variant, then ctx
  m_kvmanager->setActiveVariant(new_variant, -1);
}

size_t QnnNspModel::getEmbeddings(std::span<float> embds, InferenceStep& step) {
  qualla::Timer start;

  QnnUtils::Tensor* output_spec =
      m_nsp_graphs.back()(step.variant, step.ctx_size)
          ->getOutput(m_pooled_output ? m_layerNames[LayerType::POOL_OUTPUT]
                                      : m_layerNames[LayerType::SEQ_OUTPUT]);

  if (output_spec == nullptr) {
    __ERROR("encountered null buffer");
    throw std::runtime_error("Model is not supporting per token embedding");
  }
  const auto scale  = output_spec->quantParam[0].scale;
  const auto offset = output_spec->quantParam[0].offset;

  auto output_datatype = QnnUtils::DataType(output_spec->tensor);

  int output_bw = output_spec->dtype.bw();

  uint8_t* output_buffer = (uint8_t*)getBuffer(output_spec);

  const int return_size = m_pooled_output ? 1 : step.n_process;

  if (!m_pooled_output) {
    // If multiple tokens embedding are returned, offset to the correct location in the buffer
    if (step.variant == step.ctx_size) {
      // This was left-padded, tokens embedding are at [n_tokens - n_processed, n_tokens]
      output_buffer += (step.variant - return_size) * m_embd_size * output_bw;
    } else {
      // This was right-padded, tokens embedding are at indexes [0, n_processed]
      output_buffer += (step.n_process - 1) * m_embd_size * output_bw;
    }
  }

  const int output_len = static_cast<int>(return_size * m_embd_size);
  __TRACE("qnn-htp: get-embds for {} tokens. scale = {}, offset = {}, Returning {}",
          step.n_process,
          scale,
          offset,
          output_len);

  switch (output_datatype) {
    case QNN_DATATYPE_UFIXED_POINT_8:
      deQuantizeOutputs((uint8_t*)output_buffer, embds, scale, offset, output_len);
      break;
    case QNN_DATATYPE_UFIXED_POINT_16:
      deQuantizeOutputs((uint16_t*)output_buffer, embds, scale, offset, output_len);
      break;
    case QNN_DATATYPE_FLOAT_16:
      castOutputs((uint16_t*)output_buffer, embds, output_len, output_bw);
      break;
    case QNN_DATATYPE_FLOAT_32:
      castOutputs((float*)output_buffer, embds, output_len, output_bw);
      break;
    default:
      __ERROR("Unsupported output datatype");
  }

  __DEBUG("qnn-htp: getEmbeddings complete : {} usec (return_size={})",
          start.elapsed_usec(),
          output_len);
  return output_len;
}

size_t QnnNspModel::getIOBufferByName(std::string tensor_name, void*& buffer, bool isPrompt) {
  int32_t token;
  int32_t ctxt;
  if (isPrompt) {
    token = nsp_graph_count.rbegin()->first.first;
    ctxt  = nsp_graph_count.rbegin()->first.second;
  } else {
    token = nsp_graph_count.begin()->first.first;
    ctxt  = nsp_graph_count.begin()->first.second;
  }
  __DEBUG("getIOBufferByName isPrompt {} token {} ctxt {}", isPrompt, token, ctxt);
  for (QnnNspGraph& graph : m_nsp_graphs) {
    GraphVariant* variant = graph(token, ctxt);
    if (variant->getOutput(tensor_name) != nullptr) {
      buffer             = getBuffer(variant->getOutput(tensor_name));
      size_t buffer_size = getBufferSize(variant->getOutput(tensor_name));
      __DEBUG("qnn-htp: getIOBufferByNam output tensor_name {} address {} buffer_size {}",
              tensor_name,
              (uint64_t)buffer,
              buffer_size);

      return token;
    }
    if (variant->getInput(tensor_name) != nullptr) {
      if (tensor_name != m_layerNames[LayerType::INPUT]) {
        __ERROR("QnnNspModel getting input tensor buffer {}. cont", m_layerNames[LayerType::INPUT]);
        continue;
      }
      buffer             = getBuffer(variant->getInput(tensor_name));
      size_t buffer_size = getBufferSize(variant->getInput(tensor_name));
      __DEBUG("qnn-htp: getIOBufferByNam input tensor_name {} address {} buffer_size {}",
              tensor_name,
              (uint64_t)buffer,
              buffer_size);
      return token;
    }
  }

  return token;
}

}  // namespace qualla
