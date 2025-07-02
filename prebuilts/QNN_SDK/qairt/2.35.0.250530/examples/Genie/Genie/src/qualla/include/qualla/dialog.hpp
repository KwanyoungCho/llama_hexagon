//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_DIALOG_HPP
#define QUALLA_DIALOG_HPP

#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "qualla/DialogCallback.hpp"
#include "qualla/context.hpp"
#include "qualla/detail/exports.h"
#include "qualla/detail/gpio-marker.hpp"
#include "qualla/detail/sentence.hpp"
#include "qualla/detail/tensor.hpp"
#include "qualla/detail/trie.hpp"
#include "qualla/encoder.hpp"
#include "qualla/engine.hpp"
#include "qualla/env.hpp"
#include "qualla/sampler.hpp"
#include "qualla/tokenizer.hpp"

namespace qualla {

class Dialog : public State {
 public:
  Dialog(std::shared_ptr<Env> env, const std::string& name, const qualla::json& conf);
  virtual ~Dialog();

  // Response callback
  // Called for each decodable token.
  using Callback = std::function<bool(const std::string&, Sentence::Code)>;

  // Token-to-Embedding callback
  // Called to convert each output token into an input embedding.
  using T2ECallback = std::function<void(
      Dialog&, const int32_t token, void* embedding, const uint32_t embeddingLength)>;

  // Prime LLM for a specific context.
  QUALLA_API virtual bool prime(const std::string& str);

  // Query LLM.
  // Response is provided via Callback
  QUALLA_API virtual bool query(const std::string& str, Sentence::Code, Callback rsp);

  // Query LLM.
  // Response is provided via Callback
  QUALLA_API virtual bool query(const std::vector<uint32_t>& input,
                                Sentence::Code scode,
                                qualla::DialogCallback& callback);

  QUALLA_API virtual bool query(std::vector<uint8_t>& embedding_vectors,
                                Sentence::Code scode,
                                Dialog::T2ECallback t2eCallback,
                                Dialog::Callback callback);

  // Ask a complete question
  bool ask(const std::string& str, Callback& callback) {
    return query(str, Sentence::COMPLETE, callback);
  }

  // Reset the dialog state/history
  QUALLA_API virtual void reset();

  // Save the dialog state/history
  QUALLA_API virtual bool save(const std::string& name = "");

  // Restore the dialog state/history
  QUALLA_API virtual bool restore(const std::string& name = "");

  // Dialog KPIs
  struct KPIs {
    struct Tps {
      size_t n_prompt;
      size_t n_generate;
      float prompt;
      float generate;
      float tokenAcceptance;
    };

    Kpi init;      // init (model load, mem allocs, etc) stats
    Kpi prompt;    // prompt processor stats
    Kpi generate;  // generator stats
    Kpi save;      // save stats
    Kpi restore;   // restore stats
    Kpi lora;      // lora stats
    Tps tps{0};    // TPS for prompt, generate, etc

    KPIs() { reset(); }

    QUALLA_API void reset();  // reset to initial state

    QUALLA_API std::string dump(
        std::string_view sep = " ") const;  // dump KPIs as a formated string
  };

  // Get refs to various layers
  Context& context() { return *_ctx; }
  Tokenizer& tokenizer() { return *_tokenizer; }
  Encoder* encoder() { return _encoder.get(); }
  Sampler& sampler(const std::string& role = "primary") { return *_sampler[role]; }
  Engine& engine(const std::string& role = "primary") { return *_engine[role]; }
  bool isSamplerPresent(std::string role) { return _sampler.find(role) != _sampler.end(); }

  void requantEmbedding(void* from, void* to, size_t length);

  size_t inputBitWidth{32};
  // Get latest KPIs.
  // Updates TPS, etc as needed.
  QUALLA_API KPIs& kpis();

  // List available dialog types
  QUALLA_API static std::vector<std::string> list();

  // Dialog registration
  using Creator =
      std::function<Dialog*(std::shared_ptr<Env>, const std::string&, const qualla::json&)>;
  QUALLA_API static void __register(const std::string& type, Creator func);

  // Create Dialog instance
  QUALLA_API static std::unique_ptr<Dialog> create(std::shared_ptr<Env> env,
                                                   const std::string& name,
                                                   const qualla::json& conf = {});
  QUALLA_API static std::unique_ptr<Dialog> create(std::shared_ptr<Env> env,
                                                   const std::string& name,
                                                   std::istream& json_stream);
  QUALLA_API static std::unique_ptr<Dialog> create(std::shared_ptr<Env> env,
                                                   const std::string& name,
                                                   const std::filesystem::path& json_path);
  QUALLA_API virtual bool applyLoraAdapter(std::string lora_adapter_name, std::string engine_role);
  QUALLA_API virtual bool applyLoraStrength(std::string tensor_name,
                                            float tensor_val,
                                            std::string engine_role);

  QUALLA_API virtual int getEmbeddingLength() { return _ctx->n_embd(); };

  QUALLA_API virtual int getEmbeddingBufferSize() {
    return _engine["primary"]->getEmbeddingBufferSize();
  };

  QUALLA_API bool kvRewindPrefixMatch(std::vector<int32_t>& p_vec);

  QUALLA_API void setStopSequence(const qualla::json& newStopSeqsJson);

  QUALLA_API virtual bool setOemKey(const std::string& oemKey);

  QUALLA_API virtual bool setExecutionPriority(std::string engine_role, uint32_t exeuctionPriority);

  QUALLA_API std::shared_ptr<Env> getEnv() { return _env; };

  QUALLA_API virtual bool bindEngine(const std::string& engineRole, std::shared_ptr<Engine> engine);

  QUALLA_API std::shared_ptr<Engine> getEngine(const std::string& engineRole);

  QUALLA_API void inputTensorQuantParam(std::string& dataType,
                                        double& scale,
                                        int32_t& offset,
                                        size_t& bitWidth);

  /**
   * Perform sanity checks on the dialog.
   *
   * @throw runtime_error if an invalid configuration is identified.
   */
  QUALLA_API virtual void validate() const;

 protected:
  const std::string _type;

  std::shared_ptr<Env> _env;  // Shared between multiple dialogs
  std::unique_ptr<Context> _ctx;
  std::unique_ptr<Encoder> _encoder;
  std::unique_ptr<Tokenizer> _tokenizer;
  std::unique_ptr<GpioMarker> _gpio_marker;

  std::unordered_map<std::string, std::unique_ptr<Sampler>> _sampler;  // samplers (indexed by role)
  std::unordered_map<std::string, std::shared_ptr<Engine>> _engine;    // engines  (indexed by role)

  std::string _prompt_type;
  std::vector<std::string> _inst_tags;
  std::vector<std::string> _sys_tags;
  std::vector<std::string> _role_tags;
  std::string _sys_prompt;
  SequenceMatchTrie _stop_sequence;

  KPIs _kpis;
  uint32_t _n_queries{0};       // number of queries
  uint32_t _n_past{0};          // number of tokens cached
  uint32_t _n_prompt{0};        // number of prompt tokens    (last query)
  uint32_t _n_generated{0};     // number of generated tokens (last query)
  int32_t _last_tok{-1};        // last generated token
  bool detectedStopSeq{false};  // whether stop sequence ended the query.
  std::vector<std::string> partialStopSeqMatchTokens;
  std::vector<uint32_t> partialStopSeqMatchIndexes;
  uint32_t _n_previous_prompt{0};     // number of prompt tokens    (last query)
  uint32_t _n_previous_generated{0};  // number of generated tokens (last query)
  T2ECallback m_t2eCallback{nullptr};
  InputType m_inputType{InputType::UNKNOWN};
  bool m_rewindAtBoundary{false};

  // Process dialog input tokens
  virtual bool process(std::vector<int32_t>& tokens, Callback callback) = 0;

  virtual bool process(std::vector<int32_t>& tokens, DialogCallback callback) = 0;

  /**
   *  @return true if this dialog type has long context support.
   * */
  virtual bool supportsLongContext() const { return false; };

  // Process embedding vectors
  virtual bool process(std::vector<uint8_t>& embedding_vectors,
                       Dialog::T2ECallback t2eCallback,
                       Dialog::Callback callback) {
    throw std::runtime_error("embedding input type is not supported by dialog");
  };

  virtual bool dequantToFloat() { return false; };

  // Finds the top-k tokens from the given logits, provided that their probabilities satisfy
  // pThreshold. Since this is primarily used for multistream, each token is pushed to a separate
  // vector to generate the first token for K streams.
  void getTopK(Tensor& logits,
               std::vector<std::vector<int32_t>>& tokens,
               size_t topK,
               float pThreshold,
               Dialog::Callback callback);

  template <typename T>
  void runTopK(Tensor& logits,
               std::vector<std::vector<int32_t>>& tokens,
               size_t topK,
               float pThreshold,
               Dialog::Callback callback);

  // Set error/failed state, and dispatch the callback
  bool abort(const std::string& err, Callback& callback) {
    State::error(err);
    callback("", Sentence::ABORT);
    return false;
  }

  bool abort(const std::string& err, qualla::DialogCallback& callback) {
    State::error(err);
    callback.callBack(nullptr, 0, Sentence::ABORT, tokenizer());
    return false;
  }

  std::string accumulatePartialStopSeqMatches() {
    std::ostringstream accumulatedStr;
    for (const auto& str : partialStopSeqMatchTokens) {
      accumulatedStr << str;
    }
    return accumulatedStr.str();
  }

  void addPartialStopSeqMatches(const std::string& str, uint32_t index) {
    partialStopSeqMatchTokens.emplace_back(str);
    partialStopSeqMatchIndexes.emplace_back(index);
  }

  void clearPartialStopSeqMatches() {
    partialStopSeqMatchTokens.clear();
    partialStopSeqMatchIndexes.clear();
  }

  bool getStopSeqCallback(const std::string& str, Sentence::Code c, Dialog::Callback callback);

 private:
  std::unordered_map<std::string, std::unordered_map<std::string, Dialog::T2ECallback>>
      m_t2eCallbacks;

  void tokenToEmbedCallback(const int32_t token, void* embedding, const uint32_t embeddingSize);

  template <class F, class T>
  void tokenToEmbedRequantCallback(const int32_t token,
                                   void* embedding,
                                   const uint32_t embeddingSize);

  void calculateRequantEncodings();

  std::string inputDataType{"QNN_DATATYPE_FLOAT_32"};
  double inputScale{1.0};
  int32_t inputOffset{0};

  std::shared_ptr<void> embeddingLut;
  std::string lutDataType{"QNN_DATATYPE_FLOAT_32"};
  double lutScale{1.0};
  int32_t lutOffset{0};

  double requantScale{1.0};
  int32_t requantOffset{0};
  size_t lutByteWidth{1};
};

}  // namespace qualla

#endif  // QUALLA_DIALOG_HPP
