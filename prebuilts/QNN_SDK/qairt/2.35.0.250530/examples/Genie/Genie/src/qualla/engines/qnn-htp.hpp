//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef __QNN_HTP_H__
#define __QNN_HTP_H__

#include <fmt/format.h>

#include <string>
#include <vector>

#include "nsp-image-model.hpp"
#include "nsp-model.hpp"
#include "qualla/detail/config.hpp"
#include "qualla/detail/onload.hpp"
#include "qualla/detail/timer.hpp"
#include "qualla/engine.hpp"

namespace qualla {

class NspEngine : public Engine {
 protected:
  QnnNspBaseModel::Params _params;

  std::unique_ptr<QnnNspBaseModel> _model;
  std::vector<std::pair<uint32_t, uint32_t>> _tokensCheckpoint;
  std::unordered_map<std::string, std::vector<std::pair<uint32_t, uint32_t>>>
      _savedTokenCheckpoints;
  std::string _model_type;

  // Internal helper function to process all input types.
  size_t processAll(const std::vector<int32_t>& tokens,
                    std::vector<uint8_t>& embeddings,
                    const uint16_t* featureVector,
                    const std::vector<int32_t>& selected,
                    const int32_t start_idx,
                    const bool post_update,
                    const std::vector<int32_t>& attention_map,
                    std::vector<float>& logits,
                    bool logits_all);

  size_t processAll(const std::vector<int32_t>& tokens,
                    std::vector<uint8_t>& embeddings,
                    const uint16_t* featureVector,
                    const std::vector<int32_t>& selected,
                    const int32_t start_idx,
                    const bool post_update,
                    const std::vector<int32_t>& attention_map,
                    Tensor& logits,
                    bool logits_all);

 public:
  NspEngine(Context& ctx, const qualla::json& json);
  virtual ~NspEngine();

  virtual size_t process(const std::vector<int32_t>& tokens,
                         std::vector<float>& logits,
                         bool logits_all) override;

  virtual size_t process(const std::vector<int32_t>& tokens,
                         Tensor& logits,
                         bool logits_all) override;

  virtual size_t process(const std::vector<int32_t>& tokens,
                         const std::vector<int32_t>& attention_map,
                         std::vector<float>& logits,
                         bool logits_all) override;

  virtual size_t process(const std::vector<int32_t>& tokens,
                         const std::vector<int32_t>& attention_map,
                         Tensor& logits,
                         bool logits_all) override;

  virtual size_t process(std::vector<uint8_t>& embeddings,
                         const std::vector<int32_t>& attention_map,
                         Tensor& logits,
                         bool logits_all) override;

  virtual size_t process(std::vector<uint8_t>& embeddings,
                         const std::vector<int32_t>& attention_map,
                         std::vector<float>& logits,
                         bool logits_all) override;

  virtual size_t process(std::vector<uint8_t>& embedding_vectors,
                         const uint16_t* featureVector,
                         const std::vector<int32_t>& selected,
                         const int32_t start_idx,
                         const bool post_update,
                         const std::vector<int32_t>& attention_map,
                         std::vector<float>& logits,
                         bool logits_all) override;
  virtual size_t process(std::vector<uint8_t>& embedding_vectors,
                         const uint16_t* featureVector,
                         const std::vector<int32_t>& selected,
                         const int32_t start_idx,
                         const bool post_update,
                         const std::vector<int32_t>& attention_map,
                         Tensor& logits,
                         bool logits_all) override;
  virtual size_t process(const std::vector<uint8_t>& inputs,
                         std::vector<uint8_t>& outputs) override;

  /** Stores a precomputed EOS embedding vector. */
  virtual bool cacheEosEmbedding(std::vector<uint8_t>& eosEmbedding) override;

  void getInputQuantParam(double& scale, int& offset) { _model->getInputQuantParam(scale, offset); }

  virtual qualla::InputType getInputType() override;

  virtual void getTensorParam(LayerType layerType,
                              std::string& dataType,
                              double& scale,
                              int32_t& offset,
                              size_t& bitWidth) override;

  virtual void getTensorDimensions(LayerType layerType, std::vector<uint32_t>& dimensions) override;

  virtual size_t getEmbeddingBufferSize() override;

  virtual bool updateKV(size_t n_past) override;
  virtual bool updateKV(size_t n_past, const std::vector<bool>& selected) override;
  virtual bool save(const std::string& name) override;
  virtual bool saveKvToBuffer(Buffer* kv_buff) override;
  virtual bool getCacheSpec(CacheFileSpec& spec) override;
  virtual bool getKVHead(
      CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale) override;
  virtual size_t restore(const std::string& name, bool chooseHigherVariant) override;
  virtual void reset() override;

  virtual bool set(qualla::json data) override;
  virtual qualla::json get() override;

  virtual bool load() override;
  virtual bool unload() override;

  virtual bool applyLoraAdapter(std::string lora_adapter_name) override;
  virtual bool applyLoraStrength(std::string tensor_name, float tensor_val) override;
  virtual bool updateTokenCheckpoint(uint32_t token, uint32_t kvCacheIndx) override;
  virtual bool removeTokenCheckpoint(size_t removeAmt) override;
  virtual std::pair<uint32_t, int32_t> rewindKVCacheToPrefixMatch(std::vector<int32_t>& tokens,
                                                                  uint32_t& past) override;
  virtual bool setOemkey(const std::string& oemKey) override;
  virtual bool setExecutionPriority(const uint32_t executionPriority) override;
  virtual size_t getBuffer(void*& buffer, std::string bufferName, bool isPrompt) override;
  virtual void setSharedCounter(std::atomic<int32_t>& counter) override {
    if (_model) _model->setSharedCounter(counter);
  }
  virtual void resetSharedCounter() override {
    if (_model) _model->resetSharedCounter();
  }
  virtual std::string getTokenMapFilePath() override;

  virtual void setRunProcess(uint8_t run_process) override { _model->setRunProcess(run_process); }
  QUALLA_API virtual void updatedEmbeddingLength(uint32_t embedLength) override {
    if (_model) _model->updatedEmbeddingLength(embedLength);
  }

  QUALLA_API virtual bool isLongContextEnabled() const override;
};

}  // namespace qualla

#endif
