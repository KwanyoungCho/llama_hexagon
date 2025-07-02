//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_ENGINE_HPP
#define QUALLA_ENGINE_HPP

#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "qualla/context.hpp"
#include "qualla/detail/buffer.hpp"
#include "qualla/detail/cache-file.hpp"
#include "qualla/detail/exports.h"
#include "qualla/detail/kpi.hpp"
#include "qualla/detail/tensor.hpp"

namespace qualla {

class Engine : public State {
 public:
  QUALLA_API Engine(Context& ctx, const std::string& type, const qualla::json& conf = {});
  QUALLA_API virtual ~Engine();

  // Engine features
  struct Feature {
    enum Flags {
      OUTPUT_LOGITS     = (1UL << 0),  // Output of this engine is Logits
      OUTPUT_EMBEDDINGS = (1UL << 1),  // Output of this engine is Embeddings
      SAVE_RESTORE      = (1UL << 2),  // Save and restore support
      DYNAMIC_LOAD      = (1UL << 3)   // Dynamic loading / unloading support
    };
  };

  // Get engine feature mask
  uint32_t features() const { return _features; }
  bool supports(uint32_t flag) const { return _features & flag; }

  // Get engine type
  const std::string& type() const { return _type; }

  // Get engine role
  const std::string& role() const { return _role; }

  // Process input tokens and generate output.
  // The output is Logits for LLM and Embeddings for Sentence Transformers.
  // Returns the number of tokens in the output.
  // TODO: remove implementation here once supported by all engines
  QUALLA_API virtual size_t process(const std::vector<int32_t>& tokens,
                                    std::vector<float>& output,
                                    bool output_all = false) = 0;

  QUALLA_API virtual size_t process(const std::vector<int32_t>& tokens,
                                    Tensor& output,
                                    bool output_all = false) = 0;

  // Process input tokens and generate output.
  // The output is Logits for LLM and Embeddings for Sentence Transformers.
  // If provided, use attention mask and postion ids based on attention_map
  // Returns the number of tokens in the output.
  // TODO: remove implementation here once supported by all engines
  QUALLA_API virtual size_t process(const std::vector<int32_t>& tokens,
                                    const std::vector<int32_t>& attention_map,
                                    std::vector<float>& output,
                                    bool output_all = false);

  QUALLA_API virtual size_t process(std::vector<uint8_t>& embeddings,
                                    const std::vector<int32_t>& attention_map,
                                    std::vector<float>& output,
                                    bool output_all = false);

  QUALLA_API virtual size_t process(const std::vector<int32_t>& tokens,
                                    const std::vector<int32_t>& attention_map,
                                    Tensor& output,
                                    bool output_all = false);

  // for embedding as input
  QUALLA_API virtual size_t process(std::vector<uint8_t>& embeddings,
                                    const std::vector<int32_t>& attention_map,
                                    Tensor& output,
                                    bool output_all = false);

  // Process input tokens without returning the output.
  // Returns the number of tokens in the output.
  QUALLA_API virtual size_t process(const std::vector<int32_t>& tokens);

  QUALLA_API virtual size_t process(const std::vector<uint8_t>& inputs,
                                    std::vector<uint8_t>& outputs);

  // Synchronize the state of the context & engine.
  // n_past is the number of tokens in the KV Cache.
  QUALLA_API virtual bool updateKV(size_t n_past);
  // selected: selected tokens in the current run
  QUALLA_API virtual bool updateKV(size_t n_past, const std::vector<bool>& selected);

  QUALLA_API virtual bool save(const std::string& name);
  QUALLA_API virtual size_t restore(const std::string& name, bool chooseHigherVariant = false);
  QUALLA_API virtual bool saveKvToBuffer(qualla::Buffer* kv_buff);

  QUALLA_API virtual void reset();
  QUALLA_API virtual bool getCacheSpec(CacheFileSpec& spec);
  QUALLA_API virtual bool getKVHead(
      CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale);
  QUALLA_API virtual bool setKVHead(
      CacheFileSpec spec, uint32_t layer, uint32_t head, void* data, double* scale);

  QUALLA_API virtual bool cacheEosEmbedding(std::vector<uint8_t>& eosEmbedding);

  // Calculates the expected size of an embedding vector in bytes.
  QUALLA_API virtual size_t getEmbeddingBufferSize();

  QUALLA_API virtual qualla::InputType getInputType();

  QUALLA_API virtual void getTensorParam(
      LayerType layerType, std::string& dataType, double& scale, int32_t& offset, size_t& bitWidth);

  QUALLA_API virtual void getTensorDimensions(LayerType layerType,
                                              std::vector<uint32_t>& dimensions);
  // Load/unload all model/state data
  QUALLA_API virtual bool load();
  QUALLA_API virtual bool unload();

  // Set internal/run-time engine params/data
  QUALLA_API virtual bool set(qualla::json data);

  // Get internal/run-time engine params/data
  QUALLA_API virtual qualla::json get();

  QUALLA_API virtual size_t getBuffer(void*& buffer, std::string bufferName, bool isPrompt);

  QUALLA_API virtual size_t process(std::vector<uint8_t>& embedding_vectors,
                                    const uint16_t* featureVector,
                                    const std::vector<int32_t>& selected,
                                    const int32_t start_idx,
                                    const bool post_update,
                                    const std::vector<int32_t>& attention_map,
                                    std::vector<float>& logits,
                                    bool logits_all);
  QUALLA_API virtual size_t process(std::vector<uint8_t>& embedding_vectors,
                                    const uint16_t* featureVector,
                                    const std::vector<int32_t>& selected,
                                    const int32_t start_idx,
                                    const bool post_update,
                                    const std::vector<int32_t>& attention_map,
                                    Tensor& logits,
                                    bool logits_all);
  QUALLA_API virtual void setSharedCounter(std::atomic<int32_t>& counter);
  QUALLA_API virtual void resetSharedCounter();
  QUALLA_API virtual void setRunProcess(uint8_t run_process = 0);
  QUALLA_API virtual void updatedEmbeddingLength(uint32_t embedLength);

  QUALLA_API virtual bool isLongContextEnabled() const;

  // Engine KPIs
  struct KPIs {
    Kpi load;
    Kpi process;
    Kpi update_kv;
    Kpi unload;

    KPIs() { reset(); }

    QUALLA_API void reset();                                        // reset to initial state
    QUALLA_API std::string dump(std::string_view sep = " ") const;  // dump KPIs as formated string
  };

  // Get Engine KPIs
  KPIs& kpis() { return _kpis; }

  // Get Engine context
  Context& context() { return _ctx; }

  // List available engines
  QUALLA_API static std::vector<std::string> list();

  // Create Engine instance
  QUALLA_API static std::shared_ptr<Engine> create(Context& ctx, std::istream& json_stream);
  QUALLA_API static std::shared_ptr<Engine> create(Context& ctx, const std::string& json_str);
  QUALLA_API static std::shared_ptr<Engine> create(Context& ctx, const qualla::json& conf = {});

  // Engine registration
  using Creator = std::function<Engine*(Context&, const qualla::json&)>;
  QUALLA_API static void __register(const std::string& type, Creator func);
  QUALLA_API virtual bool applyLoraAdapter(std::string lora_adapter_name);
  QUALLA_API virtual bool applyLoraStrength(std::string tensor_name, float tensor_val);
  QUALLA_API virtual bool updateTokenCheckpoint(uint32_t token, uint32_t kvCacheIndx);
  QUALLA_API virtual bool removeTokenCheckpoint(size_t removeAmt);
  QUALLA_API virtual std::pair<uint32_t, int32_t> rewindKVCacheToPrefixMatch(
      std::vector<int32_t>& tokens, uint32_t& past);
  QUALLA_API virtual bool setOemkey(const std::string& oemKey);
  QUALLA_API virtual bool setExecutionPriority(const uint32_t executionPriority);

  QUALLA_API virtual std::string getTokenMapFilePath();
  // Engine bound
  QUALLA_API bool isBound() { return _bound; }
  QUALLA_API void bound() { _bound = true; }
  QUALLA_API void unBound() { _bound = false; }

 protected:
  std::string _type;      // engine type
  std::string _role;      // engine role
  Context& _ctx;          // reference to the context
  Env& _env;              // reference to the env
  KPIs _kpis;             // our KPIs
  uint32_t _features{0};  // engine feature mask
  std::atomic_bool _bound{false};
};

}  // namespace qualla

#endif  // QUALLA_ENGINE_HPP
