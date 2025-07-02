//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  Confidential & Proprietary - Qualcomm Technologies, Inc. ("QTI")

#ifndef __QUALLA_QNN_GPU_MODEL_H_
#define __QUALLA_QNN_GPU_MODEL_H_

#include <atomic>
#include <filesystem>
#include <string>
#include <vector>

#include "IOTensor.hpp"
#include "QnnApi.hpp"
#include "qnn-utils.hpp"
#include "qualla/detail/tensor.hpp"
#include "qualla/env.hpp"

namespace qualla {

// Maintain a list of named tensors for
static std::string INPUT_IDS = "input_ids";
static std::string ATTN_MASK = "attention_mask";
static std::string LOGITS    = "logits";
static std::string POS_IDS   = "position_ids";

class QnnGpuModel {
 public:
  struct Params {
    std::filesystem::path model_basedir;
    std::vector<std::string> model_list;  // model filenames
    uint32_t ctx_size;
    uint32_t num_heads;
    uint32_t head_dim;
    uint32_t vocab_size;
  };

  struct GpuKVCache {
    bool isKey;
    uint32_t tensorId;
    QnnUtils::Tensor* tensorUtil;

    GpuKVCache() {
      isKey      = false;
      tensorUtil = nullptr;
      tensorId   = 0;
    }
    GpuKVCache(bool _isKey, uint32_t _tensorId, QnnUtils::Tensor* _tensorUtil)
        : isKey(_isKey), tensorId(_tensorId), tensorUtil(_tensorUtil) {}
  };

  // QNN specific variables
  std::unique_ptr<QnnApi> _qnnApi;
  std::unique_ptr<IOTensor> _ioTensor{nullptr};

  // Model Location Storage
  const std::filesystem::path _modelBaseDir;
  std::vector<std::string> _modelList;
  std::map<uint32_t, std::pair<uint32_t, std::string>, std::greater<int>> _modelVariants;

  bool _useDmabufIo;

  // Model parameters
  uint32_t _ctxSize{0};
  uint32_t _numHeads{0};
  uint32_t _headDim{0};
  uint32_t _nVocabSize{0};

  // Information regarding model execution settings and last inference

  // Model specific variables
  uint32_t _numGraphs;
  // I/O Tensor Informations
  std::unordered_map<std::string, Qnn_Tensor_t*> _inputTensors;
  std::unordered_map<std::string,
                     std::unordered_map<std::string, std::shared_ptr<QnnUtils::Tensor>>>
      _inputSpecs;

  std::unordered_map<std::string, Qnn_Tensor_t*> _outputTensors;
  std::unordered_map<std::string,
                     std::unordered_map<std::string, std::shared_ptr<QnnUtils::Tensor>>>
      _outputSpecs;

  // Store Input Output Pointer for faster access
  struct IoTensorList {
    QnnUtils::Tensor* inputIds{nullptr};
    QnnUtils::Tensor* attnMask{nullptr};
    QnnUtils::Tensor* positionIds{nullptr};
    QnnUtils::Tensor* logits{nullptr};
    bool expandCausalMask;
  };
  std::unordered_map<uint32_t, IoTensorList> t_list;

  // _numTokensProcessed defines number of population of kvcache
  // _numCurrentTokensProcessed defines the number of tokens executed
  // by engine in current set of token
  size_t _numTokensProcessed{0};
  size_t _numCurrentTokensProcessed{0};

  std::vector<GpuKVCache> _kvCache;

  std::map<std::string, std::pair<double, uint16_t>> timeLogs;

  // Model Constructor
  QnnGpuModel(Env& env, const Params& params);
  ~QnnGpuModel();

  bool initializeModel(void);
  bool initializeIOTensorPerGraph(qnn_wrapper_api::GraphInfo_t* const& graphInfo,
                                  std::string& graphName,
                                  std::string sharedGraphName = "");
  bool initializeIOTensors(void);
  bool initializeTensorPointers();

  void prepareCausalMask(uint16_t* attnMaskBuffer, uint32_t currQuerySize);
  void setupInputTensors(uint32_t maxQuerySize,
                         uint32_t currQuerySize,
                         const std::vector<int32_t>& tokens,
                         const std::vector<int32_t>& attention_map);

  template <class T1, class T2>
  inline bool executeModel(T1& input, T2& output, std::string graph_name);

  size_t runInference(const std::vector<int32_t>& tokens,
                      const std::vector<int32_t>& attention_map,
                      std::vector<float>& logits,
                      bool logits_all = false);

  size_t runInference(const std::vector<int32_t>& tokens, Tensor& logits, bool logits_all = false);

  size_t loadKVCache(const std::string& save_path);
  bool saveKVCache(const std::string& load_path);
  bool reset();

 private:
  Env& _env;
  // Internal functions to separate different runInference logic
  bool runInferenceHelper(std::string& graphName,
                          int32_t* wait_time_total,
                          int32_t* exec_time_total);
  size_t processLogits(uint32_t graphVariant,
                       uint32_t currQuerySize,
                       std::vector<float>& logits,
                       bool logits_all);
  size_t processLogits(uint32_t graphVariant,
                       uint32_t currQuerySize,
                       Tensor& logits,
                       bool logits_all);
  inline void* getBuffer(QnnUtils::Tensor& spec) { return _ioTensor->getBuffer(spec.tensor); }
  inline void* getBuffer(QnnUtils::Tensor* spec) { return _ioTensor->getBuffer(spec->tensor); }
  inline size_t getBufferSize(QnnUtils::Tensor& spec) { return spec.dims.getSize(); }
  inline size_t getBufferSize(QnnUtils::Tensor* spec) { return spec->dims.getSize(); }
  inline size_t getNumElements(QnnUtils::Tensor& spec) { return spec.dims.getNumElements(); }
  inline size_t getNumElements(QnnUtils::Tensor* spec) { return spec->dims.getNumElements(); }

  // Parse KV$ Tensor names here - supports past_{key,value}_{layer_idx}[_h0]_{in,out}
  std::tuple<int, int> parseKVTensorName(std::string name);
};

}  // namespace qualla

#endif
