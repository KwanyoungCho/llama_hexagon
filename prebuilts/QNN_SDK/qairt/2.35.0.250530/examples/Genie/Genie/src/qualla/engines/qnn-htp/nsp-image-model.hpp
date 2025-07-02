//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef __QUALLA_NSP_IMAGE_MODEL_H_
#define __QUALLA_NSP_IMAGE_MODEL_H_

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
#include "qualla/detail/threadpool.hpp"
#include "qualla/env.hpp"

namespace qualla {

class QnnNspImageModel : public QnnNspBaseModel {
 protected:
  // Maps tensor name to allocation block index and block offset
  std::map<std::string, std::pair<int, size_t>> tensor_alloc_info;

  std::string embedding_datatype{"QNN_DATATYPE_FLOAT_32"};

  // Maps layers to their tensor names.
  std::map<LayerType, std::string> m_layerNames{{LayerType::INPUT, "pixel_values"},
                                                {LayerType::OUTPUT, "image_features"}};
  inline bool updateTensorPointer(GraphVariant& variant, std::string& key, QnnUtils::Tensor*& t);

 public:
  std::vector<std::string> model_filelist;
  std::vector<std::string> exec_select_graphs;
  bool load_select_graphs;

  // Model parameters
  ModelArchitectureType m_modelArchitectureType;

  QnnUtils::DataType d_input{QNN_DATATYPE_INT_32};
  QnnUtils::DataType d_output{QNN_DATATYPE_INT_32};

  // Model specific variables
  uint32_t m_num_graphs;

  bool _threaded{false};
  uint64_t _cpumask{0};
  ThreadPool threadpool;

  // Store some pointers for easier access
  QnnUtils::Tensor* t_pixel_values{nullptr};
  QnnUtils::Tensor* t_image_features{nullptr};

  QnnNspImageModel(Env& env, const QnnNspBaseModel::Params& params);

  ~QnnNspImageModel();

  bool setupInputTensors(const std::vector<uint8_t>& inputs);

  bool setupInputFP16(const std::vector<uint8_t>& inputs);

  template <typename DType>
  bool setupInput(const std::vector<uint8_t>& inputs);

  bool quantizeInput(float* in, size_t tensorOffset, size_t length);

  bool initializeModel(void);
  bool validateModel(void);
  bool initializeIOTensors(void);
  bool initializeTensorPointers();

  size_t getEmbeddingBufferSize();

  void getTensorDimensions(LayerType layerType, std::vector<uint32_t>& dimensions);

  void getTensorParam(
      LayerType layerType, std::string& dataType, double& scale, int32_t& offset, size_t& bitWidth);

  size_t runInference(const std::vector<uint8_t>& inputs, std::vector<uint8_t>& outputs);

  void setHigherVariant();
};

}  // namespace qualla

#endif