//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef __QUALLA_NSP_MODEL_H_
#define __QUALLA_NSP_MODEL_H_

#include <atomic>
#include <filesystem>
#include <span>
#include <string>
#include <vector>

#include "IOTensor.hpp"
#include "QnnApi.hpp"
#include "kvmanager.hpp"
#include "nsp-base-model.hpp"
#include "nsp-graph.hpp"
#include "qnn-utils.hpp"
#include "qualla/detail/tensor.hpp"
#include "qualla/env.hpp"

namespace qualla {

class QnnNspModel : public QnnNspBaseModel {
 protected:
  // Populated by allocateTensors()
  // Maps tensor name to allocation block index and block offset
  std::map<std::string, std::pair<int, size_t>> tensor_alloc_info;

  int32_t input_width    = 1;
  int32_t input_channel  = 1;
  int32_t input_bitWidth = 4;

  int32_t embedding_length = -1;
  std::string embedding_datatype{"float32"};

  // Maps layers to their tensor names.
  std::map<LayerType, std::string> m_layerNames{{LayerType::INPUT, "input_ids"},
                                                {LayerType::OUTPUT, "logits"},
                                                {LayerType::TOKEN_TYPE_IDS, "token_type_ids"},
                                                {LayerType::POOL_OUTPUT, "pooled_output"},
                                                {LayerType::SEQ_OUTPUT, "sequence_output"},
                                                {LayerType::ATTN_MASK, "attention_mask"},
                                                {LayerType::POS_SIN, "position_ids_sin"},
                                                {LayerType::POS_COS, "position_ids_cos"},
                                                {LayerType::POS_IDS, "position_ids"},
                                                {LayerType::ANCHOR, "anchor_buffer"},
                                                {LayerType::CACHE_INDEX, "cache_index"},
                                                {LayerType::INPUT_EMBED, "inputs_embeds"},
                                                {LayerType::VALID_MASK, "valid_token_mask"}};

  std::vector<uint8_t> m_eosEmbedding;

  Qnn_TensorDataFormat_t m_expectedDataFormat{QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER};

 public:
  std::vector<std::string> model_filelist;
  std::string lmhead_weight_dir;
  bool token_history_enabled{true};
  std::vector<int32_t> token_history;
  std::map<int32_t, int32_t> variant_latency;
  std::vector<std::string> exec_select_graphs;
  bool load_select_graphs;

  // Model parameters
  ModelArchitectureType m_modelArchitectureType;
  int32_t m_ctx_size{-1};
  int32_t m_vocab_size{-1};
  int32_t m_kv_dim{-1};
  int32_t m_embd_size{-1};
  int32_t m_pad_token{-1};

  size_t m_embeddingBufferSize{0};

  QnnUtils::DataType d_input{QNN_DATATYPE_INT_32}, d_kv{QNN_DATATYPE_UFIXED_POINT_8},
      d_attn_map{QNN_DATATYPE_UFIXED_POINT_16}, d_token_type{QNN_DATATYPE_INT_32};

  // Information regarding model execution settings and last inference
  struct RunInfo {
    int32_t n_tokens;
    size_t n_processed;

    std::vector<int32_t> tokens;
  } run_info{-1, 0, {}};

  // Model specific variables
  uint32_t m_num_graphs;
  bool _lmhead_weight_input{false};

  bool _threaded{false};
  uint64_t _cpumask{0};

  KVManagerMode _kv_update_method{SMART_MASK};
  bool m_kv_use_scatter{false};
  std::unique_ptr<KVManager> m_kvmanager;

  // Store some pointers for easier access
  QnnUtils::Tensor* t_input_ids{nullptr};
  QnnUtils::Tensor* t_cache_index{nullptr};
  QnnUtils::Tensor* t_attn_mask{nullptr};
  QnnUtils::Tensor* t_token_type_ids{nullptr};
  QnnUtils::Tensor* t_valid_mask{nullptr};

  uint32_t m_cache_position_size{0};  // Temporarily support vector cache positions

  // Variables for attention mask
  union {
    uint8_t u8;
    uint16_t u16;
    uint32_t u32;
  } m_attention_positive_value, m_attention_negative_value;

  // Variables for positional encodings
  PositionalEncoding m_positional_encoding;
  QnnUtils::DataType d_pos{QNN_DATATYPE_UFIXED_POINT_16};
  // PositionalEncodingType::ABSOLUTE OR PositionalEncodingType::ALIBI
  QnnUtils::Tensor* t_position_ids{nullptr};
  // PositionalEncodingType::ROPE variables
  int32_t m_pos_dim{-1};    // Dimension of positional embedding tensor (incl partial_factor)
  void* rope_sin{nullptr};  // Pre-calculated RoPE sin table of size [ctx_size, m_pos_dim]
  void* rope_cos{nullptr};  // Pre-calculated RoPE cos table of size [ctx_size, m_pos_dim]

  QnnUtils::Tensor* t_position_ids_sin{nullptr};
  QnnUtils::Tensor* t_position_ids_cos{nullptr};

  // Variables for long context
  LongContextParams m_longcontext;

  QnnNspModel(Env& env, const Params& params);

  ~QnnNspModel();

  bool initializeModel(void);
  bool validateModel(void);
  bool initializeIOTensors(void);
  bool initializeTensorPointers();
  bool initializeKVManager(const size_t numThreads,
                           const uint64_t cpuMask,
                           const bool enablePolling);
  bool calculate_rope_embeddings(void);
  bool load_lmhead_weight_as_input(void);

  std::vector<int32_t> parseAttentionMask(int32_t n_inputs,
                                          int32_t n_past,
                                          const std::vector<int32_t>& attention_map);

  void getInputQuantParam(double& scale, int& offset);

  size_t getIOBufferByName(std::string tensor_name, void*& buffer, bool isPrompt);

  bool setupInput(const InferenceStep& step,
                  int32_t start,
                  const std::vector<int32_t>& tokens,
                  std::vector<uint8_t>& embeddings,
                  const uint16_t* featureVector,
                  const std::vector<int32_t>& selected,
                  const int32_t start_idx,
                  const bool post_update,
                  const std::vector<int32_t>& attention_map = {},
                  int32_t map_n_past                        = 0,
                  int32_t map_n_inputs                      = 0);

  bool setupInputEmbeddings(const InferenceStep& step,
                            const bool pad_left,
                            const std::vector<uint8_t>& eagle_embed_in,
                            const uint16_t* eagle_feature_in,
                            const std::vector<int32_t>& selected,
                            const int32_t start_idx,
                            const int32_t embed_in_idx,
                            const bool post_update);

  template <typename DType>
  void setupAttentionMask(const InferenceStep& step,
                          int32_t start,
                          const std::vector<int32_t>& attention_map,
                          int32_t map_n_past,
                          int32_t map_n_inputs);

  template <typename DType>
  bool setupAlibiPositionEmbedding(const InferenceStep& step);

  bool quantizeInput(float* in, size_t tensorOffset, size_t length);

  size_t getEmbeddingBufferSize();

  size_t runInference(const std::vector<int32_t>& tokens,
                      std::vector<uint8_t>& embeddings,
                      const uint16_t* featureVector,
                      const std::vector<int32_t>& selected,
                      const int32_t start_idx,
                      const bool post_update,
                      const std::vector<int32_t>& attention_map,
                      std::vector<float>& output,
                      bool output_all = false);

  size_t runInference(const std::vector<int32_t>& tokens,
                      std::vector<uint8_t>& embeddings,
                      const uint16_t* featureVector,
                      const std::vector<int32_t>& selected,
                      const int32_t start_idx,
                      const bool post_update,
                      const std::vector<int32_t>& attention_map,
                      Tensor& output,
                      bool output_all = false);

  bool cacheEosEmbedding(std::vector<uint8_t>& eosEmbedding);

  bool setKVCacheNPast(size_t n_past, const std::vector<bool>& selected);

  size_t getEmbeddings(std::span<float> embds, InferenceStep& step);

  size_t getEmbeddings(Tensor& embds, InferenceStep& step);

  void getTensorParam(
      LayerType layerType, std::string& dataType, double& scale, int32_t& offset, size_t& bitWidth);

  size_t getDequantLogits(std::span<float> buffer, InferenceStep& step, int32_t count);

  size_t getLogits(Tensor& logits,
                   InferenceStep& step,
                   int32_t count,
                   bool requireLogitsCopy = false);

  bool debugOutputs(const InferenceStep& step, const std::string& tensor_name);

  void setHigherVariant();

  size_t loadKVCache(const std::string& load_path, bool chooseHigherVariant = false);
  bool saveKVCache(const std::string& save_path);
  bool saveKVCacheToBuffer(Buffer* kv_buff);
  bool getCacheSpec(CacheFileSpec& spec);
  bool getKVHead(CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale);

  template <typename T>
  size_t getQuantLogits(std::span<T> logits, bool logits_all);

  bool setupInputTensors(const std::vector<uint16_t*> eagle_embed_in,
                         const uint16_t* eagle_feature_in,
                         const std::vector<int32_t> selected,
                         const int32_t start_idx,
                         const bool post_update,
                         int32_t n_past,
                         std::span<const int32_t> attention_map,
                         size_t n_skip_prefix,
                         size_t n_apply_prefix_offsef);

  bool isLongContextEnabled() const override {
    return m_longcontext.mode != LongContextParams::Mode::DISABLED;
  }

  bool _skip_logits_tensor_check{false};

  // Eagle extra feature buffer
  uint16_t* eagle_extra_feature{nullptr};

  std::string draftFeatureName = "last_hidden_states";

  // for vocab trim
  bool m_vocab_trim{false};
  int32_t m_vocab_trim_size{-1};

 protected:
  inline void syncDrafTargetPrefill(bool isDraft, bool isReset);
  void updateFeatureBuffer(int32_t count);
  // Internal functions to separate different runInference logic

  inline bool updateTensorPointer(GraphVariant& variant, std::string& key, QnnUtils::Tensor*& t);
};
}  // namespace qualla
#endif
