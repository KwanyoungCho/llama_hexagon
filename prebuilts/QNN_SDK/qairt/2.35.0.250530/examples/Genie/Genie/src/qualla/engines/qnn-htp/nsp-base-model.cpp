//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#define _USE_MATH_DEFINES  // Used for M_PI

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <iostream>
#include <set>
#include <span>
#include <sstream>

#include "fmt/format.h"
#include "fmt/os.h"
#include "fmt/ranges.h"
#include "fp16/fp16.h"
#include "native-kv.hpp"
#include "nsp-base-model.hpp"
#include "qualla/detail/cache-file.hpp"
#include "qualla/detail/timer.hpp"
#include "qualla/env.hpp"
#include "smart-mask.hpp"

namespace fs = std::filesystem;

#define __ERROR(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_ERROR, fmt::format(__fmt, ##__VA_ARGS__))
#define __WARN(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_WARN, fmt::format(__fmt, ##__VA_ARGS__))
#define __INFO(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_INFO, fmt::format(__fmt, ##__VA_ARGS__))
#define __KPIS(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __DEBUG(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
#define __TRACE(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))

namespace qualla {

QnnNspBaseModel::QnnNspBaseModel(Env& env, const Params& params)
    : model_basedir(params.model_basedir), _env(env) {
  // Debug flags
  _debug_path       = params.debug_path;
  _debug_specs      = params.debug_specs;
  _debug_tensors    = params.debug_tensors;
  _debug_outputs    = params.debug_outputs;
  _debug_qnn        = params.debug_qnn;
  lora_conf         = params.lora_config_type;
  _backend_lib      = params.backend_lib;
  _backend_ext_conf = params.backend_ext_conf;

  if (lora_conf != LoraConfigType::LORA_DISABLE) {
    lora_config.insert(params.lora_param.begin(), params.lora_param.end());
  }
  lora_alpha_val = {};
  for (auto it : lora_config) {
    alpha_tensor_name = it.second.alpha_tensor_name;
    for (auto idx = 0; idx < it.second.alphas.size(); idx++) {
      lora_alpha_val[it.second.alphas[idx]] = it.second.alpha_tensor_val[idx];
    }
  }
}

bool QnnNspBaseModel::float32ToFloat16(uint8_t* out, float* in, size_t numElements) {
  if (!numElements) return false;
  uint16_t* temp = (uint16_t*)out;
  for (size_t i = 0; i < numElements; i++) {
    temp[i] = fp16_ieee_from_fp32_value(in[i]);
  }
  return true;
}

bool QnnNspBaseModel::setOemKey(const std::string& oemKey) {
  if (m_qnnApi != nullptr) {
    return m_qnnApi->setOemKey(oemKey);
  }
  return false;
}

bool QnnNspBaseModel::setExecutionPriority(const uint32_t executionPriority) {
  if (m_qnnApi != nullptr) {
    return m_qnnApi->setExecutionPriority(static_cast<Qnn_Priority_t>(executionPriority));
  }
  return false;
}

bool QnnNspBaseModel::flushLoraWeightsBuffers(void) {
  if (!_lora_enabled) {
    __ERROR("qnn-htp: Model does not support LoRA weights.");
    return false;
  }

  for (auto& variant : m_variant_list) {
    for (auto& [tname, tspec] : variant.input_specs) {
      if (tname.find("lora") !=
          std::string::npos) {  // find lora weights tensors and flush them out
        if (getBuffer(tspec) == nullptr) return false;
        size_t numElements = tspec.dims.getNumElements();
        auto offset        = tspec.quantParam[0].offset;
        // Since values needs to be quantized so zero is going to get translated.
        switch (tspec.dtype) {
          case QNN_DATATYPE_UFIXED_POINT_8:
            std::fill_n((uint8_t*)getBuffer(tspec), numElements, static_cast<uint8_t>(-offset));
            break;
          case QNN_DATATYPE_UFIXED_POINT_16:
            std::fill_n((uint16_t*)getBuffer(tspec), numElements, static_cast<uint16_t>(-offset));
            break;
          case QNN_DATATYPE_FLOAT_16: {
            uint16_t* buffer = (uint16_t*)getBuffer(tspec);
            for (int i = 0; i < numElements; i++) {
              buffer[i] = fp16_ieee_from_fp32_value(-offset);
            }
            break;
          }
          default:
            __ERROR("Unsupported {} datatype for {} tensor", tspec.dtype.str(), tname);
            return false;
        }
      }
    }
  }
  return true;
}

bool QnnNspBaseModel::applyLoraWeights(const std::string& lora_weights_name) {
  if (!_lora_enabled) {
    __ERROR("qnn-htp: Model does not support LoRA weights.");
    return false;
  }
  if (lora_conf != LoraConfigType::LORA_INPUT_WEIGHT_ENABLE) {
    __ERROR("qnn-htp: LoRA config is not enable for input weights");
    return false;
  }

  if (!lora_config.contains(lora_weights_name)) {
    __ERROR("qnn-htp: Could not find lora weights config to apply ");
    return false;
  }

  if (_lora_enabled && lora_config[lora_weights_name].path.empty()) {
    __ERROR("qnn-htp: LoRA weights dir is empty for {}", lora_weights_name);
    return false;
  }

  adapter = lora_weights_name;
  for (auto idx = 0; idx < lora_config[lora_weights_name].alpha_tensor_val.size(); idx++) {
    if (!applyLoraStrength(lora_config[lora_weights_name].alphas[idx],
                           lora_alpha_val[lora_config[lora_weights_name].alphas[idx]])) {
      __ERROR("qnn-htp: Could not apply Alpha tensor ");
      return false;
    }
  }

  for (auto& variant : m_variant_list) {
    for (auto& [tname, tspec] : variant.input_specs) {
      if (tname.find("lora") != std::string::npos &&
          tname != lora_config[lora_weights_name].alpha_tensor_name) {
        if (getBuffer(tspec) == nullptr) return false;
        // lora tensor file names should be in same format as tensor names present in graph
        std::string lora_weights_file =
            (model_basedir / fs::path(lora_config[lora_weights_name].path) /
             fs::path(tname + ".raw"))
                .string();

        size_t numElements = tspec.dims.getNumElements();
        auto scale         = tspec.quantParam[0].scale;
        auto offset        = tspec.quantParam[0].offset;

        size_t size = sizeof(float);
        std::vector<float> lora_weights_f32;  // Temporary variable to load fp32 values
        lora_weights_f32.reserve(numElements);

        FILE* fp = fopen(lora_weights_file.c_str(), "r");
        if (fp == NULL) {
          __ERROR("NSPModel: Error opening file: {}", lora_weights_file);
          return false;
        }

        size_t count = fread(lora_weights_f32.data(), size, numElements, fp);
        fclose(fp);

        if (count != numElements) {
          __ERROR("NSPModel: Could not load {} - expected file size {}",
                  lora_weights_file,
                  numElements * size);
          return false;
        }

        // Quantize the values
        switch (tspec.dtype) {
          case QNN_DATATYPE_UFIXED_POINT_8:
            QnnUtils::quantizeTensorPtr(
                lora_weights_f32.data(), (uint8_t*)getBuffer(tspec), offset, scale, numElements);
            break;
          case QNN_DATATYPE_UFIXED_POINT_16:
            QnnUtils::quantizeTensorPtr(
                lora_weights_f32.data(), (uint16_t*)getBuffer(tspec), offset, scale, numElements);
            break;
          case QNN_DATATYPE_FLOAT_16:
            float32ToFloat16((uint8_t*)getBuffer(tspec), lora_weights_f32.data(), numElements);
            break;
          default:
            __ERROR("Unsupported {} datatype for {} tensor", tspec.dtype.str(), tname);
            return false;
        }
      }
    }
  }
  return true;
}

bool QnnNspBaseModel::applyBinarySections(std::vector<std::string>& binsection_list) {
  if (graph_switching && lazy_lora == "lazy") {
    m_qnnApi->m_adapterCache.clear();
  }

  // apply binary section for lora config
  for (int i = 0; i < binsection_list.size(); i++) {
    if (binsection_list.at(i).empty()) continue;
    __DEBUG("qnn-htp: applyBinarySections adapters {}", binsection_list.at(i));
    if (!m_qnnApi->applyBinarySection(i, binsection_list.at(i), m_use_mmap, graph_switching, lazy_lora)) {
      __ERROR("qnn-htp: Error in applyBinarySections {}", i);
      return false;
    }
  }
  return true;
}

bool QnnNspBaseModel::applyLoraStrength(const std::string& alpha_name, const float alpha_val) {
  if (alpha_tensor_name.empty() || alpha_name.empty()) return true;

  bool alphaFound = false;
  for (auto it : lora_config) {
    auto itt = std::find(it.second.alphas.begin(), it.second.alphas.end(), alpha_name);
    if (itt != it.second.alphas.end()) {
      lora_alpha_val[alpha_name] = alpha_val;
      alphaFound                 = true;
    }
  }
  if (!alphaFound) {
    __ERROR("qnn-htp: Could not find lora alpha tensor to apply");
    return false;
  }

  if (!adapter.empty()) {
    for (auto idx = 0; idx < lora_config[adapter].alphas.size(); idx++) {
      lora_config[adapter].alpha_tensor_val[idx] = lora_alpha_val[lora_config[adapter].alphas[idx]];
    }
  } else {
    // Alpha tensor gets set (below) when adapter is applied.
    return true;
  }

  for (auto& variant : m_variant_list) {
    if (!variant.input_specs.contains(alpha_tensor_name)) continue;

    auto& tspec          = variant.input_specs.at(alpha_tensor_name);
    auto [scale, offset] = tspec.quantParam[0];

    switch (tspec.dtype) {
      case QNN_DATATYPE_UFIXED_POINT_8:
        QnnUtils::quantizeTensorPtr(lora_config[adapter].alpha_tensor_val.data(),
                                    (uint8_t*)getBuffer(tspec),
                                    offset,
                                    scale,
                                    lora_config[adapter].alpha_tensor_val.size());
        break;
      case QNN_DATATYPE_UFIXED_POINT_16:
        QnnUtils::quantizeTensorPtr(lora_config[adapter].alpha_tensor_val.data(),
                                    (uint16_t*)getBuffer(tspec),
                                    offset,
                                    scale,
                                    lora_config[adapter].alpha_tensor_val.size());
        break;
      case QNN_DATATYPE_FLOAT_16:
        float32ToFloat16((uint8_t*)getBuffer(tspec),
                         const_cast<float*>(lora_config[adapter].alpha_tensor_val.data()),
                         lora_config[adapter].alpha_tensor_val.size());
        break;
      default:
        __ERROR("Unsupported alpha tensor dtype {}", tspec.dtype.str());
        return false;
    }
    __DEBUG("qnn-htp: applyAlphaTensor alpha = {}", alpha_val);
    return true;  // Each lora bin section should have only one alpha tensor
  }
  return false;
}

bool QnnNspBaseModel::applyLoraAdapter(const std::string& lora_adapter_name) {
  if (lora_conf != LoraConfigType::LORA_ADAPTER_WEIGHT_ENABLE) {
    __ERROR("qnn-htp: Lora config is not enable for adapters");
    return false;
  }

  if (!lora_config.contains(lora_adapter_name)) {
    __ERROR("qnn-htp: Could not find lora adapters config to apply ");
    return false;
  }

  adapter = lora_adapter_name;
  for (auto idx = 0; idx < lora_config[lora_adapter_name].alpha_tensor_val.size(); idx++) {
    if (!applyLoraStrength(lora_config[lora_adapter_name].alphas[idx],
                           lora_alpha_val[lora_config[lora_adapter_name].alphas[idx]])) {
      __ERROR("qnn-htp: Could not apply Alpha tensor ");
      return false;
    }
  }

  if (!applyBinarySections(lora_config[lora_adapter_name].binsection_list)) {
    __ERROR("qnn-htp: Could not apply binary Sections ");
    return false;
  }

  for (auto& variant : m_variant_list) variant.refreshTensorQuantParams();

  return true;
}

// Dumps out the specified tensor to _debug_path numbered according to m_inference_count
bool QnnNspBaseModel::debugOutputs(const InferenceStep& step, const std::string& tensor_name) {
  GraphVariant* graph_variant = m_nsp_graphs.back()(step.variant, step.ctx_size);
  QnnUtils::Tensor* tensor    = graph_variant->getOutput(tensor_name);
  if (tensor == nullptr) {
    __DEBUG("qnn-htp: Couldn't find tensor {} in graph {}", tensor_name, graph_variant->graph_name);
    return false;
  }

  const int output_bitwidth = tensor->dtype.bw();  // Detect 8-bit vs 16-bit logits
  uint32_t rank             = tensor->tensor->v1.rank;
  size_t numElements        = 1;
  for (int i = 0; i < rank; i++) {
    numElements *= tensor->tensor->v1.dimensions[i];
  }
  const int32_t output_size = output_bitwidth * numElements;
  std::string fname = fmt::format("{}/{}/{:03d}", _debug_path, tensor_name, m_inference_count);
  if (!QnnUtils::writeRawData(getBuffer(tensor), output_size, fname)) {
    __DEBUG("qnn-htp: Failed to save {}. Error when writing to {}", tensor_name, fname);
    return false;
  }
  return true;
}

void QnnNspBaseModel::dumpTensorSpecs() {
  if (!fs::exists(_debug_path) && !fs::create_directories(_debug_path)) {
    __ERROR("Could not create directory for debug - {}", _debug_path);
    return;
  }

  const uint32_t n_graphs                     = m_qnnApi->getGraphsCount();
  qnn_wrapper_api::GraphInfo_t**& graphs_info = m_qnnApi->getGraphsInfo();

  static const char* stringFmt =
      "\t\t{ \"name\": \"%s\", \"dims\": [%d, %d, %d, %d], \"bitwidth\": %d, "
      "\"dtype\": \"%s\", \"dataFormat\": %u, \"scale\": [%s], \"offset\": [%s] },\n";
  for (auto graph_idx = 0; graph_idx < n_graphs; graph_idx++) {
    qnn_wrapper_api::GraphInfo_t* const graph_info = graphs_info[graph_idx];

    // Create output spec file and open it
    std::string filename = fmt::format("{}/spec.{}.json", _debug_path, graph_info->graphName);

    FILE* specFile = fopen(filename.c_str(), "w");
    if (specFile == NULL) throw std::runtime_error("Error opening file : " + filename);

    fprintf(specFile, "{\n\t\"graph_name\" : \"%s\",\n", graph_info->graphName);
    for (bool io : {true, false}) {
      uint32_t n_tensors   = (io) ? graph_info->numInputTensors : graph_info->numOutputTensors;
      Qnn_Tensor_t* tensor = (io) ? graph_info->inputTensors : graph_info->outputTensors;

      fprintf(specFile, (io) ? "\t\"inputs\" : [\n" : "\t\"outputs\" : [\n");
      while (n_tensors-- > 0) {
        std::string tname                         = QnnApi::getTensorName(*tensor);
        const auto [t, dims, quant_params, dtype] = QnnUtils::Tensor(tensor);
        auto& [b, h, w, c, bw]                    = dims;

        std::string scales, offsets;
        QnnUtils::getQuantParamString(quant_params, scales, offsets);

        fprintf(specFile,
                stringFmt,
                tname.c_str(),
                b,
                h,
                w,
                c,
                bw,
                dtype.str(),
                QNN_TENSOR_GET_DATA_FORMAT(tensor),
                scales.c_str(),
                offsets.c_str());
        tensor++;
      }
      fseek(specFile, -2, SEEK_CUR);  // Remove trailing comma
      fprintf(specFile, "\n\t],\n");
    }
    fseek(specFile, -2, SEEK_CUR);  // Remove trailing comma
    fprintf(specFile, "\n}");
    fclose(specFile);
  }
}


}  // namespace qualla
