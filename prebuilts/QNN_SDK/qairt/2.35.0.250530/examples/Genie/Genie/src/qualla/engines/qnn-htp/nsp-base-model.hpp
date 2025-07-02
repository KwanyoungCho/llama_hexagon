//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef __QUALLA_NSP_BASE_MODEL_H_
#define __QUALLA_NSP_BASE_MODEL_H_

#include <atomic>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "IOTensor.hpp"
#include "QnnApi.hpp"
#include "kvmanager.hpp"
#include "nsp-graph.hpp"
#include "qnn-utils.hpp"
#include "qualla/detail/tensor.hpp"
#include "qualla/env.hpp"

namespace qualla {

enum ModelArchitectureType : uint8_t { DECODER = 0, ENCODER = 1 };

enum LoraConfigType : uint8_t {
  LORA_DISABLE               = 0,
  LORA_INPUT_WEIGHT_ENABLE   = 1,
  LORA_ADAPTER_WEIGHT_ENABLE = 2
};

static const std::unordered_set<Qnn_DataType_t> supported_activations = {
    QNN_DATATYPE_UFIXED_POINT_8,
    QNN_DATATYPE_UFIXED_POINT_16,
    QNN_DATATYPE_INT_32,
    QNN_DATATYPE_FLOAT_32,
    QNN_DATATYPE_FLOAT_16};

class QnnNspBaseModel : public State {
 public:
  const std::filesystem::path model_basedir;
  struct LoraConfig {
    std::string lora_name;
    std::vector<std::string> binsection_list;  // loarv2 adapter bins filenames
    std::string path;                          // lorav1 weights directory.
    std::string alpha_tensor_name;             // lorav2 alpha tensor names
    std::vector<std::string> alphas;           // lorav2 alpha names
    std::vector<float> alpha_tensor_val;       // lorav2 alpha tensor values
  };
  struct Params {
    ModelArchitectureType modelArchitectureType;  // Model architecture
    std::filesystem::path model_basedir;          // model basedir
    std::vector<std::string> model_list;          // model filenames
    std::map<int32_t, int32_t> variant_latency;   // latency for different variants
    std::vector<std::string> exec_select_graphs;  // Execute selected graphs
    bool load_select_graphs;  // Load only graphs mentioned in exec_select_graphs from the context
                              // bin, by default all graphs are loaded

    bool use_mmap;
    uint64_t data_alignment_size;
    bool use_async_Init;
    uint64_t mmap_budget;
    int64_t spill_fill_bufsize;
    int32_t ctx_size;
    int32_t kv_dim;
    int32_t pad_token;
    size_t n_embd;
    uint32_t n_threads{0};
    uint64_t cpumask{0};
    bool poll{false};
    std::string backend_lib;
    std::string backend_ext_conf;
    std::string debug_path;
    std::string draft_tok_map;
    bool debug_specs;
    bool debug_tensors;
    bool debug_outputs;
    bool debug_qnn;
    std::string kv_update_method;
    std::string lmhead_weight_dir;
    bool graph_switching;
    std::string lazy_lora;
    LoraConfigType lora_config_type;
    std::map<std::string, LoraConfig> lora_param;
    std::string input_layer_name;
    int32_t embedding_length;
    std::string embedding_datatype;
    bool pooled_output;
    bool disable_kv_cache;
    // Parameters for positional encodings
    PositionalEncoding positional_encoding_params;

    // Parameters for long context
    LongContextParams longcontext_params;
  };

  enum RUN_PROCESS : uint8_t {
    OVERALL_PROCESS = 0,
    PART_RUN        = 1,
    NO_RUN_LMHEAD   = 2
  } _run_process = OVERALL_PROCESS;

  QnnNspBaseModel(Env& env, const QnnNspBaseModel::Params& params);
  virtual ~QnnNspBaseModel() = default;

  virtual bool initializeModel(void)      = 0;
  virtual bool validateModel(void)        = 0;
  virtual bool initializeIOTensors(void)  = 0;
  virtual bool initializeTensorPointers() = 0;
  virtual bool initializeKVManager(const size_t numThreads,
                                   const uint64_t cpuMask,
                                   const bool enablePolling) {
    return true;
  };
  virtual bool calculate_rope_embeddings(void) { return true; };
  virtual bool load_lmhead_weight_as_input(void) { return true; };

  virtual size_t loadKVCache(const std::string& load_path, bool chooseHigherVariant = false) {
    return 0;
  };
  virtual void setHigherVariant() { return; };
  virtual bool saveKVCache(const std::string& save_path) { return true; };
  virtual bool saveKVCacheToBuffer(Buffer* kv_buff) { return true; };
  virtual bool getCacheSpec(CacheFileSpec& spec) { return true; };
  virtual bool getKVHead(
      CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) {
    return true;
  };

  virtual size_t getEmbeddingBufferSize() { return 0; };

  virtual size_t runInference(const std::vector<int32_t>& tokens,
                              std::vector<uint8_t>& embeddings,
                              const uint16_t* featureVector,
                              const std::vector<int32_t>& selected,
                              const int32_t start_idx,
                              const bool post_update,
                              const std::vector<int32_t>& attention_map,
                              std::vector<float>& output,
                              bool output_all = false) {
    return 0;
  }

  virtual size_t runInference(const std::vector<int32_t>& tokens,
                              std::vector<uint8_t>& embeddings,
                              const uint16_t* featureVector,
                              const std::vector<int32_t>& selected,
                              const int32_t start_idx,
                              const bool post_update,
                              const std::vector<int32_t>& attention_map,
                              Tensor& output,
                              bool output_all = false) {
    return 0;
  }

  virtual size_t runInference(const std::vector<uint8_t>& inputs, std::vector<uint8_t>& outputs) {
    return 0;
  }

  virtual void getInputQuantParam(double& scale, int& offset){};

  virtual bool cacheEosEmbedding(std::vector<uint8_t>& eosEmbedding) { return true; };

  virtual bool setKVCacheNPast(size_t n_past, const std::vector<bool>& selected) { return true; };

  virtual void getTensorParam(LayerType layerType,
                              std::string& dataType,
                              double& scale,
                              int32_t& offset,
                              size_t& bitWidth){};

  virtual void getTensorDimensions(LayerType layerType, std::vector<uint32_t>& dimensions){};

  void setSharedCounter(std::atomic<int32_t>& counter) { _counter = &counter; };
  void resetSharedCounter() { _counter = nullptr; };

  // Self-Specualtive Decoding
  // This prefix is not for input tokens, but just for speical tokens
  // Only the special tokens from the offset should attend the kv prefix
  int32_t _size_to_skip_kv_prefix{0};
  int32_t _offset_to_apply_kv_prefix{0};

  std::atomic<int32_t>* _counter{nullptr};

  InputType m_inputType{InputType::UNKNOWN};

  // LoRA params and configs
  std::map<std::string, float> lora_alpha_val;
  std::string adapter;
  std::string alpha_tensor_name;
  LoraConfigType lora_conf;
  std::map<std::string, LoraConfig> lora_config;
  bool _lora_enabled{false};

  // QNN specific variables
  const bool m_sharedBuffer{true};
  std::unique_ptr<QnnApi> m_qnnApi;
  std::unique_ptr<IOTensor> m_ioTensor{nullptr};
  int64_t spill_fill_buffer_size;
  bool m_use_mmap{false};
  uint32_t m_dataAlignmentSize{0};
  bool m_use_async_Init{true};
  uint64_t mmap_budget;
  bool graph_switching{false};
  std::string lazy_lora{""};
  size_t n_embd;

  bool m_pooled_output{true};
  bool m_disableKvCache{false};

  std::string _backend_lib;
  std::string _backend_ext_conf;
  std::string m_draft_tok_map;

  // Debug mode settings
  bool _debug_specs{false};
  bool _debug_tensors{false};
  bool _debug_outputs{false};
  bool _debug_qnn{false};
  std::string _debug_path;

  // Keep track of inference count
  int m_inference_count = 0;

  // QnnNspGraph contains all GraphVariants for a specific split (with index=split_idx)
  std::vector<QnnNspGraph> m_nsp_graphs;
  // GraphVariant represents one input size within one split (e.g. KV$_split_1)
  std::vector<GraphVariant> m_variant_list;

  // For ease of usage: Map from graph name to the corresponding GraphVariant
  std::unordered_map<std::string, GraphVariant*> m_graph_map;
  // This map records how many graphs have been loaded for a particular input size and context size
  std::map<std::pair<int32_t, int32_t>, int32_t> nsp_graph_count;  // [variant, ctx_size] -> count

  int32_t m_embedding_length{-1};

  // Base functionality
  bool setExecutionPriority(const uint32_t executionPriority);
  bool setOemKey(const std::string& oemKey);
  bool flushLoraWeightsBuffers(void);
  bool applyLoraStrength(const std::string& alpha_name, const float alpha_val);
  bool applyLoraWeights(const std::string& lora_weights_name);
  bool applyBinarySections(std::vector<std::string>& binsection_list);
  bool applyLoraAdapter(const std::string& lora_adapter_name);

  void setRunProcess(uint8_t run_process) { _run_process = static_cast<RUN_PROCESS>(run_process); }
  void updatedEmbeddingLength(uint32_t embedLength) { m_embedding_length = embedLength; };

  virtual bool isLongContextEnabled() const { return false; }

  bool debugOutputs(const InferenceStep& step, const std::string& tensor_name);
  void dumpTensorSpecs();
  virtual size_t getIOBufferByName(std::string tensor_name, void*& buffer, bool isPrompt) {
    return 0;
  }

 protected:
  Env& _env;

  bool float32ToFloat16(uint8_t* out, float* in, size_t numElements);
  // Internal functions to separate different runInference logic
  inline void* getBuffer(QnnUtils::Tensor& spec) { return m_ioTensor->getBuffer(spec.tensor); }
  inline void* getBuffer(QnnUtils::Tensor* spec) { return m_ioTensor->getBuffer(spec->tensor); }
  inline size_t getBufferSize(QnnUtils::Tensor& spec) { return spec.dims.getSize(); }
  inline size_t getBufferSize(QnnUtils::Tensor* spec) { return spec->dims.getSize(); }

  template <typename U, typename T>
  inline void deQuantizeOutputs(
      U* inputs, std::span<T>& outputs, const double scale, const int32_t offset, const int count) {
#pragma clang loop vectorize(enable) interleave(enable)
    for (int i = 0; i < count; ++i) outputs[i] = ((T)inputs[i] + offset) * scale;
  }

  template <typename U, typename T>
  inline void castOutputs(U* inputs,
                          std::span<T>& outputs,
                          const int numElements,
                          const int bitWidth) {
    if (bitWidth == 2) {
#pragma clang loop vectorize(enable) interleave(enable)
      for (int i = 0; i < numElements; ++i) outputs[i] = fp16_ieee_to_fp32_value(inputs[i]);
    } else if (bitWidth == 4) {
#pragma clang loop vectorize(enable) interleave(enable)
      for (size_t i = 0; i < numElements; i++) {
        outputs[i] = inputs[i];
      }
    }
  }
};
}  // namespace qualla

#endif