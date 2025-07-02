//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_DETAIL_EAGLET_HPP
#define QUALLA_DETAIL_EAGLET_HPP

#include <fmt/format.h>
#include <fmt/ranges.h>

#include <algorithm>
#include <fstream>
#include <functional>
#include <memory>
#include <qualla/detail/onload.hpp>
#include <span>
#include <string>
#include <vector>

#include "qualla/detail/json.hpp"
#include "qualla/dialog.hpp"
#include "qualla/env.hpp"

namespace fs = std::filesystem;

namespace qualla {

using qc = qualla::Config;
// template <typename T>
using Logits = Tensor;  // std::vector<T>;

class TokenTree {
 public:
  std::vector<int32_t> m_tokens;        // Tokens in this buffer
  std::vector<int32_t> m_attentionMap;  // Attention map corresponding to the tokens
  Logits m_logits;
  std::vector<float> m_probs;  // Probabliities of each token
  uint32_t m_numTokens{0};

  void add(int32_t token, int32_t attentionMap, float prob) {
    m_tokens.push_back(token);
    m_attentionMap.push_back(attentionMap);
    m_probs.push_back(prob);
    m_numTokens++;
  }

  void clear() {
    m_tokens.clear();
    m_attentionMap.clear();
    m_probs.clear();
    //   m_logits.clear();
  }
};
/**
 * @brief Manages the draft state, including sequence tracking and branching
 */
// template <typenameT>
class DraftStateManager {
 public:
  /**
   * @brief Represents the state of a single draft sequence
   */
  struct SeqDraft {
    bool isActive{false};
    bool isDrafting{false};
    bool skip{false};
    int end_idx{false};  // Ending index of this sequence among current draft tokens
    std::vector<int32_t> batchDraftOverallIdxs;
    std::vector<int32_t> targetBatchIndices;
    std::vector<int32_t> tokens;
    std::vector<float> cumulativeProbabilities;
    int32_t draftNextIdx{0};
    int32_t draftCurrentIdx{0};
    std::vector<int32_t> batchDraftNextIndices;
  };
  int m_maxParallelSequencesAllowed{0};
  int m_numDrafted{0};             // total number of tokens drafted (pending acceptance/rejection)
  int m_numNextDraftedTokens{0};   // total number of tokens drafted for expanding
  int m_numPastDraft{0};           // Number of tokens fully processed by the draft model
  int m_numPastTarget{0};          // Number of tokens fully processed by the target model
  int m_numCurrSeq{0};             // Current number of active sequences
  std::vector<SeqDraft> m_drafts;  // Stores information about the candidate sequences being drafted
  TokenTree targetTokens;
  TokenTree draftTokens;
  TokenTree nextDraftTokens;
  void reset() {
    m_drafts.clear();
    targetTokens.clear();
    draftTokens.clear();
    nextDraftTokens.clear();
  }
  ~DraftStateManager() {
    targetTokens.clear();
    draftTokens.clear();
    nextDraftTokens.clear();
    m_drafts.clear();
  };
};

/**
 * @brief Structure to hold the configuration and state of EagleDialog
 *
 */
struct EagleDialogConfig {
  int32_t draftLength;
  int32_t eos{-1};
  std::string special_eos;
  int32_t contextSize;
  int32_t maxSeqAllowed;
  int32_t numBranches;  // Branching factor at a node
  int32_t probsPerDraft;
  int32_t maxTargetTokens;

  bool draftingKvCache;
  bool vocabTrim;

  int32_t trimmedVocabSize;
  std::vector<int32_t> draftTokenMap;

  int32_t embeddingLength;
};

class EagleDialog : public Dialog {
 public:
  std::string draftEmbedBuffName    = "_model_embedding_concat_Concat_Concat_output_0";
  std::string draftFeatureBuffName  = "last_hidden_states";
  std::string targetEmbedBuffName   = "_model_embed_tokens_Gather_Gather_output_0";
  std::string targetFeatureBuffName = "last_hidden_states";
  Sampler& _d_sampler;  // Draft sampler
  Sampler& _t_sampler;  // Target sampler
  bool process(std::vector<int32_t>& tokens, Dialog::Callback callback) override;
  bool process(std::vector<uint8_t>& embedding_vectors,
               T2ECallback t2eCallback,
               Dialog::Callback callback) override;

  inline int32_t sampleTargetToken(Logits& logits) {
    int32_t id = _t_sampler.process(logits);
    return id;
  }

  inline int32_t sampleDraftToken(Logits& logits) {
    int32_t id = _d_sampler.process(logits);
    return id;
  }

  inline bool tokensToembedding(std::vector<int32_t>& tokens,
                                std::vector<uint8_t>& embedding,
                                uint16_t embedbuffsize) {
    for (auto& token : tokens) {
      std::vector<uint8_t> curTokenEmbedding(embedbuffsize, 0);
      m_t2eCallback(*this, token, curTokenEmbedding.data(), embedbuffsize);
      embedding.insert(embedding.end(), curTokenEmbedding.begin(), curTokenEmbedding.end());
    }
    return true;
  }
  inline bool tokenToembedding(int32_t token,
                               std::vector<uint8_t>& embedding,
                               uint16_t embedbuffsize) {
    std::vector<uint8_t> curTokenEmbedding(embedbuffsize, 0);
    m_t2eCallback(*this, token, curTokenEmbedding.data(), embedbuffsize);
    return true;
  }

  template <typename T>
  inline std::string vector_to_string(const std::vector<T>& vec) {
    std::string ret_str = "[";

    for (const auto n : vec) {
      ret_str += std::to_string(n) + ",";
    }

    ret_str += "]";
    return ret_str;
  }

  EagleDialog(std::shared_ptr<Env> env, const std::string& name, const json& conf);

  /**
   * @brief Part of the process of accepting tokens from a draft model and integrating them with a
   * target model. Works by evaluating & sampling tokens from the target model, comparing them with
   * tokens in the draft & updating the states of both target and draft models accordingly.
   * @param draftEngine
   * @param targetEngine
   * @param acceptedTokensFromDraft
   * @return List of accepted tokens
   */
  std::vector<int32_t> acceptFromTree(Engine* draftEngine,
                                      Engine* targetEngine,
                                      int32_t& acceptedTokensFromDraft);

  /**
   * @brief Generates and manages a draft token tree.
   *
   * Performs token drafting for multiple sequences at each level of a token tree. It samples
   * candidate tokens for each sequence, calculates probabilities, and maintains state of the draft
   * tree across multiple branches. Follows these main steps:
   * 1. Iterate over levels of the draft tree
   * 2. For each sequence in the current level:
   *    - Compute logits & generate token candidates.
   *    - Split sequence into multiple branches
   *    - Update probabilities & store the generated token
   * 3. Filter & retain top-k sequences based on probability
   * 4. Update attention maps & pass token embeddings to the engine for processing
   * @param draftEngine
   * @param targetEngine
   */
  void createDraftTokenTree(Engine* draftEngine, Engine* targetEngine);
  /**
   *
   * @param maxLength
   */
  void pruneDraftTokenTree(int32_t maxLength);

  /**
   * @brief Evaluates draft token tree by embedding and processing target tokens,
   *        then updating the draft state by removing the first token from active drafts
   * @param targetEngine
   */
  void evaluateDraftTokenTree(Engine* targetEngine);
  virtual bool process(std::vector<int32_t>& tokens, DialogCallback callback) override;
  /*
  This function required to override the quant settings
  It helps in parity with Output.
  */

 private:
  EagleDialogConfig _config;
  DraftStateManager _draftStateManager;
  std::unordered_map<int32_t, std::vector<uint8_t>> tokEmbedMap;
  uint8_t promptVariant{128};

  uint64_t draftSampleTime{0};
  uint32_t draftSampleCount{0};
  uint32_t embedBuffSize{0};

  void initializeEagleDialogConfig(const json& conf);

  bool sampleFromTargetModel(int32_t currDraftLevelIdx,
                             int32_t longestMatchedSequenceIdx,
                             int32_t& tokenIdx,
                             int32_t& token,
                             bool& hasEos);
  bool checkDraftMatch(int32_t draftTokenIdx,
                       int32_t targetTokenId,
                       int32_t& longestMatchedSequenceIdx);

  bool updateKvCache(Engine* draftEngine,
                     Engine* targetEngine,
                     int32_t& targetTokenIdx,
                     int32_t& numMatchedTokens);

  int32_t processFeatureVectors(Engine* targetEngine,
                                Engine* draftEngine,
                                int32_t longestMatchedSequenceIdx,
                                int32_t numMatchedTokens,
                                std::vector<int32_t>& acceptedTokenIds,
                                std::vector<int32_t>& selectedIndices);

  void resetAfterAcceptingFromTree(int32_t id);

  void resetDraftSkipFlags();

  /**
   * @brief Samples token candidates & computes their probabilities for a given sequence
   * @param seq The sequence inedx for which token candidates are sampled.
   * @return A pair containing the sampled token candidates & their probabilities
   */
  std::pair<std::vector<int32_t>, std::vector<float>> sampleTokenCandidates(int seq);

  /**
   * @brief Splits a sequence into multiple branches for parallel token sampling
   *
   * @param seq The sequence index to split
   * @param numBranches The number of branches to create
   * @return A vector containing the indices of the split sequences
   */
  std::vector<int> splitSequenceIntoBranches(int seq);

  /**
   * @brief Updates draft tree & target tokens with sampled tokens & probabilities
   *
   * @param seq The sequence index being updated
   * @param tokenCandidates The sampled token candidates
   * @param probs The computed probabilities for the sampled tokens
   * @param currentLevelProbabilities The cumulative probabilities for all sequences at the current
   * level
   * @param sequenceArray The array of sequence indices for each branch
   */
  void updateDraftAndTargetTokens(int32_t seq,
                                  const std::vector<int32_t>& tokenCandidates,
                                  const std::vector<float>& probs,
                                  std::vector<float>& currentLevelProbabilities,
                                  const std::vector<int>& sequenceArray,
                                  int32_t idx_tgt_parent,
                                  int32_t idx_dft_parent);

  /**
   * @brief Calculates the top-k threshold for cumulative probabilities
   * @param currentLevelProbabilities The cumulative probabilities for all sequences at the current
   * level
   * @param topK The number of top k sequences to select
   * @return The threshold value for the top-k probabilies
   */
  float calculateTopKThreshold(std::vector<float>& currentLevelProbabilities, int32_t topK);

  /**
   * @brief Marks eligible sequences for the next level based on the top-K threshold
   * @param topKThreshold The minimum cumulative probability to qualify for the next level
   * @param currentDraftTokens Tokens at the current level that pass the threshold
   */
  void markEligibleSequences(float topKThreshold, std::vector<int32_t>& currentDraftTokens);

  /**
   * @brief Update attention map for the next level based on past drafts
   * @param startDraftIdx THe starting index for the current level
   * @param numTokensCurrentLevel Number of tokens at the current level
   * @param numDraftsPast Total number of drafts processed in previous levels
   * @param newAttentionMap THe new attention map to be updated
   * @param selectedIndicesPerLevel Selected indices for attending past KV
   */
  void updateAttentionMap(int32_t startDraftIdx,
                          int32_t numTokensCurrentLevel,
                          int32_t numDraftsPast,
                          std::vector<int32_t>& newAttentionMap,
                          std::vector<int32_t>& selectedIndicesPerLevel);

  /**
   * @brief Marks eligible sequences for the next level based on the top-K threshold
   * @param embedIn  embedding buffer to be filled in
   * @param acceptedTokenIds accepted tokens, embedding vector should be accordingly
   */
  void copyEmbeddingBuffer(void* targetEmbeddingBuffer,
                           std::vector<uint8_t>& embedIn,
                           std::vector<int32_t>& acceptedTokenIds,
                           std::vector<int32_t>& selectedIndices);

  /**
   *
   * @param draftFeatureBuffer
   * @param currentDraftTokens
   * @param selectedIndicesPerLevel
   * @param startIdxOffset
   * @param numPastDraft
   * @param newAttentionMap
   */

  // embed_tokens_quantized is not implemented though
  // What's with the juggling between embeddingBuffer and embeddingInput?
  // I used embeddingInput and got rid of embeddingBuffer
  void processDraftTokens(Engine* draftEngine,
                          Engine* targetEngine,
                          void* draftFeatureBuffer,
                          std::vector<int32_t>& currentDraftTokens,
                          std::vector<int32_t>& selectedIndicesPerLevel,
                          int32_t& startIdxOffset,
                          int32_t& numPastDraft,
                          std::vector<int32_t>& newAttentionMap);

  /**
   * @brief: Returns a vector that stores all the draft tokens and their global probabilities
   */
  std::vector<std::pair<int32_t, float>> prepareTokenProbs();

  /**
   * @brief Prunes target tokens based on their probabilities, retaining only those above a given
   * threshold. Tokens with probabilities equal to the threshold are pruned if they exceed the
   * maximum allowed count.
   * @param maxLength
   * @param probThr
   * @return
   */
  std::vector<size_t> pruneTargetTokens(int32_t maxLength, float probThr);

  /**
   * @brief Updates the sequence drafts by pruning tokens with cumulative probabilities below a
   * given threshold. After pruning, it adjusts the target batch indices to account for the removed
   * elements
   * @param probThr
   * @param indicesToPrune
   */
  void updateSequenceDraft(float probThr, std::vector<size_t>& indicesToPrune);

  void loadDraftTokenMap();

  /**
   * @brief Remove element from a vector based on the specified indices
   * @param vec
   * @param indicesToPrune
   */
  template <typename T>
  void removeElements(std::vector<T>& vec, const std::vector<size_t>& indicesToPrune);
  void clearDraftTree();
  virtual void reset() override;
  virtual bool bindEngine(const std::string& engineRole, std::shared_ptr<Engine> engine) override;
};

}  // namespace qualla
#endif  // QUALLA_DETAIL_EAGLET_HPP