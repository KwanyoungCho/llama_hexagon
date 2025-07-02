//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef __QUALLA_QNN_CPU_MODEL_H_
#define __QUALLA_QNN_CPU_MODEL_H_

#include <filesystem>
#include <string>
#include <vector>

#include "IOTensor.hpp"
#include "QnnApi.hpp"
#include "qnn-utils.hpp"
#include "qualla/detail/cache-file.hpp"
#include "qualla/detail/tensor.hpp"
#include "qualla/env.hpp"

#define LLAMA_MODEL

namespace qualla {

enum LoraConfigType {
  LORA_DISABLE               = 0,
  LORA_INPUT_WEIGHT_ENABLE   = 1,
  LORA_ADAPTER_WEIGHT_ENABLE = 2
};

class QnnCpuModel {
  enum ExecutionMode { AUTODETECT, BERT_KV, KV_ONLY, BERT_ONLY };

  Env& _env;

 public:
  enum ModelInput { TOKENS = 0x01, INPUT_EMBEDDINGS = 0x02, UNKNOWN = 0xFF };
  enum ModelOutput { LOGITS = 0x0, EMBEDDINGS = 0x1 };

  struct LoraConfig {
    std::string lora_name;
    std::vector<std::string> binsection_list;  // loRAv2 adapter bins filenames
    std::string alpha_tensor_name;             // loRAv2 alpha tensor names
    std::vector<std::string> alphas;           // loRAv2 alpha names
    std::vector<float> alpha_tensor_val;       // loRAv2 alpha tensor values
  };

  struct Params {
    std::filesystem::path model_basedir;
    std::string op_package;
    std::string backend_lib;
    std::string model_bin_path;
    std::string model;
    ModelInput model_input;
    ModelOutput model_output;
    std::string embedding_datatype;
    bool use_mmap;
    uint32_t ctx_size;
    uint32_t n_threads;
    size_t n_vocab_size;
    uint32_t n_logits;
    uint32_t n_layer;
    uint32_t n_embd;
    uint32_t n_heads;
    uint32_t n_kv_heads;
    bool kv_quant;
    LoraConfigType lora_config_type;
    std::map<std::string, LoraConfig> lora_config;
  };

  const std::filesystem::path model_basedir;
  std::vector<std::string> filename_list;
  std::vector<std::string> model_order;
  std::vector<std::string> bert_model_order;
  std::vector<std::string> kv_model_order;

  std::string op_package;
  std::string backend_lib;
  std::string model_bin_path;
  std::string model;

  long long int spill_fill_buffer_size;

  std::unordered_map<std::string, Qnn_ContextHandle_t> model_context;
  ModelInput model_input;
  ModelOutput model_output;
  std::string embedding_datatype{"float32"};
  std::map<std::string, std::pair<double, uint16_t>> timeLogs;
  std::unique_ptr<QnnApi> m_qnnApi;
  std::unique_ptr<IOTensor> m_ioTensor{nullptr};

  // Model parameters

  size_t m_ctx_size{1024};
  size_t m_num_layer{0};
  size_t m_embd{0};
  size_t m_num_heads{0};
  size_t m_num_kv_heads{0};
  size_t m_head_dim{0};
  size_t m_num_tokens{0};
  std::string position_id_path_cos;
  std::string position_id_path_sin;
  int32_t eos_token_id;
  int32_t m_num_threads;
  int32_t m_numLogits;
  size_t m_vocab_size{32000};  // todo:update vocab size from tokenzier
  bool m_use_mmap{false};
  bool m_kv_quant{false};
  std::vector<uint32_t> m_kv_dim;
  std::vector<uint32_t> m_input_dim;
  std::vector<uint32_t> m_kv_scale_dim;
  std::vector<uint32_t> m_output_dim;
  std::vector<Qnn_Param_t> m_params;
  ExecutionMode m_mode{ExecutionMode::AUTODETECT};
  size_t m_embeddingBufferSize{0};

  // LoRA params and configs
  std::map<std::string, float> m_lora_alpha_val;
  std::string m_adapter;
  LoraConfigType m_loraConfigType;
  std::map<std::string, LoraConfig> m_loraConfig;

  // Save some information about the last inference run
  struct PreviousRunInfo {
    bool was_bert_mode;
    size_t num_tokens_processed;
    bool was_logits_all;
  } prev_run{false, 0};

  // Model specific variables
  uint32_t m_num_graphs;
  std::unordered_map<std::string, Qnn_Tensor_t*> m_input_tensors;
  std::unordered_map<std::string, std::unordered_map<std::string, QnnUtils::Tensor>> m_input_specs;

  std::unordered_map<std::string, Qnn_Tensor_t*> m_output_tensors;
  std::unordered_map<std::string, std::unordered_map<std::string, QnnUtils::Tensor>> m_output_specs;

  // Store some pointers for easier access
  QnnUtils::Tensor* t_logits;
  QnnUtils::Tensor* t_output_n_past;
  QnnUtils::Tensor* t_input_ids;
  QnnUtils::Tensor* t_input_ids_num_token;
  QnnUtils::Tensor* t_input_ids_reset_kvcache;
  QnnUtils::Tensor* t_input_ids_k_cache;
  QnnUtils::Tensor* t_input_ids_v_cache;
  QnnUtils::Tensor* t_input_ids_k_scale;
  QnnUtils::Tensor* t_input_ids_v_scale;
  QnnUtils::Tensor* t_input_ids_n_past;
  QnnUtils::Tensor* t_input_lora_alpha;
  float* dequant_logits_ptr{nullptr};

  // Store pointers for bert
  QnnUtils::Tensor* b_logits;
  QnnUtils::Tensor* b_input_ids;
  QnnUtils::Tensor* b_attn_mask;

#ifdef LLAMA_MODEL
  // LLama specific variables
  uint16_t position_id_dims;  // Derived from model in initializeTensorPointers
  // uint16_t position_ids_sin[1024][64];
  // uint16_t position_ids_cos[1024][64]; // RoPE Embedding tensors. Loaded from datafile
  std::unique_ptr<uint16_t[]> position_ids_sin;  // Initialized in load_precomputed_position_ids
  std::unique_ptr<uint16_t[]> position_ids_cos;  // Initialized in load_precomputed_position_ids

  QnnUtils::Tensor* t_position_ids_sin;
  QnnUtils::Tensor* t_position_ids_cos;
#else
  QnnUtils::Tensor* t_position_ids;
#endif

  // n_past defines number of population of kvcache
  size_t m_nPast{0};

  // Keep track of inference count
  int m_inference_count = 0;

  QnnCpuModel(Env& env, const Params& params);
  ~QnnCpuModel();

  bool initializeModel(void);
  bool validateModel(void);
  bool initializeIOTensors(void);
  bool initializeTensorPointers();

  void setupInputTensors(const std::vector<int32_t>& tokens, bool run_bert_mode);
  void setupInputTensors(std::vector<uint8_t>& embeddings, bool run_bert_mode);

  template <class T1, class T2>
  inline bool executeModel(T1& input, T2& output, std::string graph_name);

  void dumpTensors(std::string graph_name, bool dump_input);
  void dumpTensorSpecs();

  void printFinalLogs();

  size_t getEmbeddingBufferSize() { return m_embeddingBufferSize; }

  bool runInference(const std::vector<int32_t>& tokens, bool logits_all);
  bool runInference(std::vector<uint8_t>& embeddings, bool logits_all);
  bool setKVCacheNPast(size_t n_past);

  size_t getDequantLogits(std::vector<float>& logits, bool logits_all = false);
  size_t getLogits(Tensor& logits, bool logits_all = false);

  size_t loadKVCache(const std::string& save_path);
  bool saveKVCache(const std::string& load_path);
  bool setKVQuantHead(CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale);
  bool setKVHead(CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale);

  bool applyLoraStrength(const std::string& alpha_tensor_name, const float alpha_val);
  bool applyLoraAdapter(const std::string& lora_adapter_name);
  bool applyBinarySections(std::vector<std::string>& binsection_list);

 private:
  bool m_mmap_context_bins = false;  // mmap context binary files instead of reading them in memory
  // Internal functions to separate different runInference logic
  bool runInferenceHelper(std::vector<std::string>& exec_models,
                          int32_t* wait_time_total,
                          int32_t* exec_time_total,
                          bool pipeline_kv_update,
                          size_t update_size);

  inline void* getBuffer(QnnUtils::Tensor& spec) { return m_ioTensor->getBuffer(spec.tensor); }
  inline void* getBuffer(QnnUtils::Tensor* spec) { return m_ioTensor->getBuffer(spec->tensor); }
  inline size_t getBufferSize(QnnUtils::Tensor& spec) { return spec.dims.getSize(); }
  inline size_t getBufferSize(QnnUtils::Tensor* spec) { return spec->dims.getSize(); }
  // TODO: Seems to be some issue with m_ioTensor->getBufferSize when sharing buffers
};

}  // namespace qualla

#endif  // __QUALLA_QNN_CPU_MODEL_HPP_
