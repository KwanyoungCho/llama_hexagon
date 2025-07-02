//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include "IOTensor.hpp"
#include "QnnApi.hpp"
#include "qnn-utils.hpp"
#include "qualla/env.hpp"

namespace qualla {

enum class GraphType { NONE, DEFAULT, LUT, DECODER, LMHEAD, IMAGE_ENCODER };

inline std::string getGraphTypeStr(GraphType type) {
  if (type == GraphType::NONE)
    return "NONE";
  else if (type == GraphType::DEFAULT)
    return "DEFAULT";
  else if (type == GraphType::LUT)
    return "LUT";
  else if (type == GraphType::DECODER)
    return "DECODER";
  else if (type == GraphType::LMHEAD)
    return "LMHEAD";
  else if (type == GraphType::IMAGE_ENCODER)
    return "IMAGE_ENCODER";
  return "ERROR: GraphType not found";
}

struct GraphVariant {
  int32_t n_tokens;
  int32_t ctx_size{-1};
  std::string graph_name;

  // Graph Type
  GraphType variantType{GraphType::NONE};
  // QNN API specific variables
  qnn_wrapper_api::GraphInfo_t* graph_info;
  Qnn_ContextHandle_t context_handle;

  QnnUtils::TensorMap input_specs;
  QnnUtils::TensorMap output_specs;

  std::map<LayerType, std::string>& m_layerNames;

  Env& _env;

  GraphVariant() = delete;
  GraphVariant(qnn_wrapper_api::GraphInfo_t* g_info,
               Qnn_ContextHandle_t qnn_ctx,
               std::map<LayerType, std::string>& layerNames,
               Env& env);
  QnnUtils::Tensor* getTensor(const std::string& tensor_name) {
    QnnUtils::Tensor* ret = getInput(tensor_name);
    return (ret != nullptr) ? ret : getOutput(tensor_name);
  }
  QnnUtils::Tensor* getInput(const std::string& tensor_name) {
    return input_specs.contains(tensor_name) ? &input_specs.at(tensor_name) : nullptr;
  }
  QnnUtils::Tensor* getOutput(const std::string& tensor_name) {
    return output_specs.contains(tensor_name) ? &output_specs.at(tensor_name) : nullptr;
  }

  bool refreshTensorQuantParams();

 private:
  int32_t determineGraphInputSize();
  int32_t determineGraphContextSize();
  GraphType determineGraphType();
};

/**
 * The idea behind QnnNspGraph is to represent "common" graphs
 * For instance, both BERT-mode and KV$-mode are the same graph with different input sizes
 * QnnNspGraph will contain and manage both BERT-split-n and KV$mode-split-n
 * I/O tensors are mostly shared between these graphs, and can be managed collectively
 */
class QnnNspGraph {
 private:
  int _idx;
  Env& _env;

  // Useful pointers for graph execution (managed by NSPModel)
  QnnApi* g_qnn_api;
  IOTensor* g_buffer_mgr;

  int32_t run_wait_time, run_exec_time;  // Add more stats into a struct

  // Debug mode settings
  bool _debug_specs{false};
  bool _debug_tensors{false};
  std::string _debug_path;

 public:
  int32_t _counter{-1};

  // TODO: Remove this reference
  std::map<std::string, std::pair<int, size_t>>* tensor_alloc_info;

  // Keys represent input_id size (1<=input_size<=ctx_size)
  // Values are graph description for that input_id size
  std::map<std::pair<int32_t, int32_t>, GraphVariant*> variants;

  // Graph Type
  GraphType m_graphType{GraphType::NONE};

  QnnNspGraph(int idx, Env& env, QnnApi* qnnApi, IOTensor* ioTensor);
  ~QnnNspGraph();
  int idx() { return _idx; }
  bool addGraph(GraphVariant* graph_spec);
  void printAvailableConfigs();

  // Overload the () operator to access a [variant, ctx_size (or -1 for global match)]
  GraphVariant* operator()(const int32_t variant, const int32_t ctx_size) {
    return variants.contains({variant, ctx_size}) ? variants.at({variant, ctx_size})
                                                  : variants.at({variant, -1});
  }

  bool execute(const int32_t n_tokens, const int32_t ctx_size, const int n_inference, bool graphSwitch, std::string& lazyLora);

  void setDebugMode(bool debug_specs, bool debug_tensors, std::string debug_path) {
    _debug_path    = debug_path;
    _debug_specs   = debug_specs;
    _debug_tensors = debug_tensors;
  }
  void dumpTensors(GraphVariant* const variant, bool mode, int n_inference) const;
};

}  // namespace qualla
