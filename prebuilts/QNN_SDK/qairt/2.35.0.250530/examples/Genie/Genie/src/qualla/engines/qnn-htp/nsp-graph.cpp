//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <regex>
#include <sstream>

#include "fmt/format.h"
#include "fmt/ranges.h"
#include "nsp-graph.hpp"
#include "nsp-model.hpp"
#include "qualla/detail/timer.hpp"

// Copied from threadpool.cpp
#if defined(_WIN32)
#define NOGDI
#include "windows.h"

static int sched_yield(void) {
  Sleep(0);
  return 0;
}
#else
#include <sched.h>
#endif

// Safe limit is there to ensure In AR-N, N number or
// any other smaller number in graph name shouldn't be get selected as context length.
#define CONTEXT_SAFE_LIMIT 501

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

// GraphVariant is a self-contained graph. Represents one specific QNN Model
GraphVariant::GraphVariant(qnn_wrapper_api::GraphInfo_t *g_info,
                           Qnn_ContextHandle_t qnn_ctx,
                           std::map<LayerType, std::string> &layerNames,
                           Env &env)
    : graph_name(g_info->graphName),
      graph_info(g_info),
      context_handle(qnn_ctx),
      m_layerNames(layerNames),
      _env(env) {
  // TRACE("Parsing %s with ctx_size %d", this->graph_name.c_str(), n_ctx);

  for (bool io : {true, false}) {
    uint32_t n_tensors   = (io) ? graph_info->numInputTensors : graph_info->numOutputTensors;
    auto tensor_wrappers = (io) ? graph_info->inputTensors : graph_info->outputTensors;
    auto &tensor_specs   = (io) ? input_specs : output_specs;
    for (size_t tensor_idx = 0; tensor_idx < n_tensors; tensor_idx++) {
      Qnn_Tensor_t &tensor      = tensor_wrappers[tensor_idx];
      std::string tensor_name   = QnnApi::getTensorName(tensor);
      tensor_specs[tensor_name] = QnnUtils::Tensor(&tensor);
    }
  }

  ctx_size    = determineGraphContextSize();
  variantType = determineGraphType();
  if (variantType != GraphType::IMAGE_ENCODER) {
    n_tokens = determineGraphInputSize();
  }
  __INFO("graphName {} and its variant Type {}", graph_name, getGraphTypeStr(variantType));
}

// Attempt to determine input size from purely graph IO and context size
// Try different types of the Input to determine(for some split some these inputs are not valid)
// Try to find using 1. input_ids/input_embeds, 2. attention_mask, 3. past_key/value tensors, 4.
// logits
int32_t GraphVariant::determineGraphInputSize() {
  QnnUtils::Tensor *tensor;

  // Recognize KeyDiff scorer network. It must have an "anchor" tensor as input
  // If it has past_keys + new_keys as input, we can determine it's variant. Else, it is invariant
  if (getInput("anchor")) {
    tensor = getInput("new_keys");
    return static_cast<int32_t>((tensor) ? tensor->dims.channel : -1);
  }

  tensor = getInput(m_layerNames[LayerType::INPUT]);
  if (tensor) {
    // input_embeds -> [1, 1, AR-N, embd_size]
    // input_ids -> [1, 1, 1, AR-N]
    return static_cast<int32_t>(
        (m_layerNames[LayerType::INPUT] == "input_embeds" ||
         m_layerNames[LayerType::INPUT] == "_model_embed_tokens_Gather_Gather_output_0")
            ? tensor->dims.getNumElements() / tensor->dims.getMaxDim()
            : tensor->dims.getNumElements());
  }

  tensor = getInput(m_layerNames[LayerType::ATTN_MASK]);
  if (tensor) {
    // attention_mask -> [1, 1, AR-N, context_size]
    return static_cast<int32_t>(tensor->dims.getNumElements() / tensor->dims.getMaxDim());
  }

  // Use past_key_out tensor to find input size
  // The last dimension of past_key_out tensor will always be the input size
  for (auto &[tname, qtensor] : output_specs) {
    if (!tname.starts_with("past_key")) continue;
    return static_cast<int32_t>(qtensor.dims.channel);
  }

  tensor = getOutput(m_layerNames[LayerType::OUTPUT]);
  if (tensor) {
    // logits -> [1, 1, AR-N, vocab_size]
    return static_cast<int32_t>(tensor->dims.getNumElements() / tensor->dims.getMaxDim());
  }

  __DEBUG(
      "Couldn't determine input token length from tensors. Attempting to parse input token "
      "length "
      "from "
      "graph name.");
  // In the worst case, try to use the graph name to determine input token length
  static const std::regex pattern(R"((ar|AR)_?(\d+))");  // Match AR_32/ar_32/AR32/ar32
  std::smatch matches;
  if (std::regex_search(graph_name, matches, pattern) && matches.size() == 3)
    return std::stoi(matches[2].str());

  throw std::runtime_error("Unexpected model. Couldn't determine required input tokens " +
                           graph_name);
}

// Attempt to determine context size from purely graph IO
// The easiest way is using attention_mask. Else, past key/value can also be used
int32_t GraphVariant::determineGraphContextSize() {
  QnnUtils::Tensor *tensor;

  // Recognize KeyDiff scorer network. It must have a "score" tensor as output
  tensor = getOutput("score");
  if (tensor) return static_cast<int32_t>(tensor->dims.channel);

  tensor = getInput(m_layerNames[LayerType::ATTN_MASK]);
  if (tensor) return static_cast<int32_t>(tensor->dims.channel);
  // Use past_key_in and past_key_out tensor to find context size
  // The last dimension of past_key_in + past_key_out tensor will always be the context size
  for (auto &[tname, qtensor] : output_specs) {
    if (!tname.starts_with("past_key")) continue;
    auto in_name   = tname.substr(0, tname.rfind("_")).append("_in");
    auto in_tensor = getInput(in_name);
    if (in_tensor) return static_cast<int32_t>(qtensor.dims.channel + in_tensor->dims.channel);
  }

  __DEBUG(
      "Couldn't determine context length from tensors. Attempting to parse context length from "
      "graph name.");
  // In the worst case, try to use the graph name to determine context length
  static const std::regex pattern(R"((cl|CL)_?(\d+))");  // Match cl_1024/CL_1024/cl1024/CL1024
  std::smatch matches;
  if (std::regex_search(graph_name, matches, pattern) && matches.size() == 3)
    return std::stoi(matches[2].str());

  static const std::regex npattern(R"((\d+))");  // find all the numbers in the graph name and take
                                                 // the max number.e.g. llama3_8b_ar1_1024_1_of_7
  int mNum   = 0;
  auto begin = std::sregex_iterator(graph_name.begin(), graph_name.end(), npattern);
  auto end   = std::sregex_iterator();
  for (std::sregex_iterator i = begin; i != end; ++i) {
    const std::smatch nmatch = *i;
    int number               = std::stoi(nmatch.str());
    mNum                     = std::max(mNum, number);
  }
  if (mNum != 0 && mNum > CONTEXT_SAFE_LIMIT) return mNum;
  return -1;
}

// Classify graphs as per:
//  only input_id classify as LUT
//  past_key/value tensors are present classify as DECODER
//  Only LOGITS exists then classify as LMHEAD
//  Default for all other types of cases
GraphType GraphVariant::determineGraphType() {
  bool inputIDExists = false, pastKVExists = false, logitsExists = false,
       imageFeaturesExists = false;

  if (getInput(m_layerNames[LayerType::INPUT])) inputIDExists = true;

  // Use past_key_out tensor to find input size
  for (auto &[tname, qtensor] : output_specs) {
    if (tname.starts_with("past_key") || tname.starts_with("past_value")) pastKVExists = true;
    if (tname.starts_with("image_features")) imageFeaturesExists = true;
    if (pastKVExists || imageFeaturesExists) break;
  }

  // Use past_key_in tensor
  for (auto &[tname, qtensor] : input_specs) {
    if (tname.starts_with("past_key") || tname.starts_with("past_value")) pastKVExists = true;
    if (pastKVExists) break;
  }
  if (getOutput(m_layerNames[LayerType::OUTPUT])) logitsExists = true;

  if (inputIDExists && !pastKVExists && !logitsExists)
    return GraphType::LUT;
  else if (!inputIDExists && pastKVExists && !logitsExists)
    return GraphType::DECODER;
  else if (!inputIDExists && !pastKVExists && logitsExists)
    return GraphType::LMHEAD;
  else if (imageFeaturesExists)
    return GraphType::IMAGE_ENCODER;
  else
    return GraphType::DEFAULT;
}
bool GraphVariant::refreshTensorQuantParams() {
  for (bool io : {true, false}) {
    uint32_t n_tensors   = (io) ? graph_info->numInputTensors : graph_info->numOutputTensors;
    auto tensor_wrappers = (io) ? graph_info->inputTensors : graph_info->outputTensors;
    auto &tensor_specs   = (io) ? input_specs : output_specs;
    for (size_t tensor_idx = 0; tensor_idx < n_tensors; tensor_idx++) {
      Qnn_Tensor_t &tensor    = tensor_wrappers[tensor_idx];
      std::string tensor_name = QnnApi::getTensorName(tensor);
      std::vector<QnnUtils::QuantParam> quantParams;
      if (!QnnApi::getTensorQuantParams(&tensor, quantParams)) {
        quantParams.emplace_back(0, 0);
      }
      tensor_specs[tensor_name].quantParam = quantParams;
    }
  }
  return true;
}

QnnNspGraph::QnnNspGraph(int idx, Env &env, QnnApi *qnnApi, IOTensor *ioTensor)
    : _idx(idx), _env(env), g_qnn_api(qnnApi), g_buffer_mgr(ioTensor) {}

QnnNspGraph::~QnnNspGraph() { __DEBUG("qnn-htp: del-NSP-graph"); }

// Parse a loaded GraphInfo_t
bool QnnNspGraph::addGraph(GraphVariant *graph_spec) {
  const int32_t variant  = graph_spec->n_tokens;
  const int32_t ctx_size = graph_spec->ctx_size;

  // The variants map is keyed as [variant, ctx_size]
  const std::pair<int32_t, int32_t> key = {variant, ctx_size};
  if (variants.contains(key)) {
    const std::string &g1 = graph_spec->graph_name;
    const std::string &g2 = variants.at(key)->graph_name;
    __ERROR("qnn-htp: Detected duplicate AR-{} CL-{} graphs: {} and {}", variant, ctx_size, g1, g2);
    throw std::runtime_error("qnn-htp: duplicate graph found, likely overflow occured");
  }
  if (m_graphType == GraphType::NONE) {
    m_graphType = graph_spec->variantType;
  } else if (m_graphType != graph_spec->variantType) {
    __ERROR("qnn-htp: Detected different variant type {} for AR-{} CL-{} than graph type {}",
            getGraphTypeStr(graph_spec->variantType),
            variant,
            ctx_size,
            getGraphTypeStr(m_graphType));
    throw std::runtime_error("qnn-htp: different graph types encountered between variants.");
  }
  variants[key] = graph_spec;

  return true;
}

void QnnNspGraph::dumpTensors(GraphVariant *const variant, bool mode, int n_inference) const {
  if (n_inference >= 10) return;

  QnnUtils::TensorMap &tensor_specs = (mode) ? variant->input_specs : variant->output_specs;
  std::string prefix = fmt::format("{}/{}/{:03d}", _debug_path, variant->graph_name, n_inference);
  for (const auto &[tname, tspec] : tensor_specs) {
    std::string fname = fmt::format("{}_{}_{}", prefix, (mode) ? "in" : "out", tname);
    QnnUtils::writeRawData(g_buffer_mgr->getBuffer(tspec.tensor), tspec.dims.getSize(), fname);
  }
}

bool QnnNspGraph::execute(const int32_t n_tokens, const int32_t ctx_size, const int n_inference, bool graphSwitch, std::string& lazyLora) {
  // Allow either {variant, ctx_size} OR a global {variant, -1}
  const std::pair<int32_t, int32_t> key = {n_tokens,
                                           variants.contains({n_tokens, ctx_size}) ? ctx_size : -1};

  if (!variants.contains(key)) {
    __ERROR("Could not find AR-{} CL-{} for execution", n_tokens, ctx_size);
    return false;
  }

  GraphVariant *variant = variants.at(key);  // Assume n_tokens/ctx_size exists
  std::map<std::string, std::pair<double, uint16_t>> timeLogs;
  qnn_wrapper_api::GraphInfo_t *const graph = variant->graph_info;
  if (_debug_tensors) dumpTensors(variant, true, n_inference);  // Dump input tensors
  __DEBUG("Executing graph {} - {}", _idx, graph->graphName);

  // lazily apply binary section immediately before graph execution
  auto graphHandle = graph->graph;
  if (graphSwitch && lazyLora == "lazy" && g_qnn_api->m_adapterCache.count(graphHandle) != 0 && !std::get<3>(g_qnn_api->m_adapterCache[graphHandle])) {
    if (!g_qnn_api->applyCachedAdapter(graphHandle)) {
      __ERROR("Could not Apply Cached Adapter for graph {} - {}", _idx, graph->graphName);
      return false;
    }
  }

  if (!g_qnn_api->graphExecute(
          graph->inputTensors, graph->outputTensors, graph->graphName, timeLogs)) {
    __ERROR("qnn-htp: graph-exec failed for graph {} - {}", _idx, graph->graphName);
    return false;
  }

  if (_debug_tensors) dumpTensors(variant, false, n_inference);  // Dump output tensors
  return true;
}
}  // namespace qualla
