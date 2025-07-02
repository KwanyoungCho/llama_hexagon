//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_ENCODER_HPP
#define QUALLA_ENCODER_HPP

#include <atomic>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "qualla/context.hpp"
#include "qualla/detail/exports.h"
#include "qualla/detail/sentence.hpp"
#include "qualla/engine.hpp"
#include "qualla/env.hpp"
#include "qualla/tokenizer.hpp"

namespace qualla {

class Encoder : public State {
 public:
  QUALLA_API Encoder(std::shared_ptr<Env> env, const std::string& name, const qualla::json& conf);
  QUALLA_API virtual ~Encoder();

  // Encoder registration
  using Creator =
      std::function<Encoder*(std::shared_ptr<Env>, const std::string&, const qualla::json&)>;
  QUALLA_API static void __register(const std::string& type, Creator func);

  // Create Encoder instance
  QUALLA_API static std::unique_ptr<Encoder> create(std::shared_ptr<Env> env,
                                                    const std::string& name,
                                                    const qualla::json& conf = {});
  QUALLA_API static std::unique_ptr<Encoder> create(std::shared_ptr<Env> env,
                                                    const std::string& name,
                                                    std::istream& json_stream);
  QUALLA_API static std::unique_ptr<Encoder> create(std::shared_ptr<Env> env,
                                                    const std::string& name,
                                                    const std::filesystem::path& json_path);

  // Encode sentence
  QUALLA_API virtual bool encode(const std::string& str, std::vector<uint8_t>& output);

  QUALLA_API virtual bool encode(const std::vector<int32_t>& tokens, std::vector<uint8_t>& output);

  QUALLA_API virtual size_t getEmbeddingLutSize();

  QUALLA_API virtual void* getEmbeddingLut();
  QUALLA_API virtual int32_t getLastToken();

  // Encode image
  QUALLA_API virtual bool encode(std::vector<uint8_t>& pixel_values,
                                 std::vector<uint8_t>& image_features);

  // Get output dimensions
  QUALLA_API virtual void output_dimensions(std::vector<std::uint32_t>& outputDimensions);

  QUALLA_API virtual void outputTensorQuantParam(std::string& dataType,
                                                 double& scale,
                                                 int32_t& offset,
                                                 size_t& bitWidth);

  QUALLA_API virtual bool applyLoraAdapter(std::string lora_adapter_name,
                                           std::string engine_role = "primary");
  QUALLA_API virtual bool applyLoraStrength(std::string tensor_name,
                                            float tensor_val,
                                            std::string engine_role = "primary");

  QUALLA_API std::string type() { return _type; }

  QUALLA_API std::shared_ptr<Env> getEnv() { return _env; };

  // Embedding KPIs
  struct KPIs {
    struct Tps {
      size_t n_prompt;
      float prompt;
    };

    Kpi init;    // init (model load, mem allocs, etc) stats
    Kpi prompt;  // prompt processor stats
    Kpi lora;    // lora stats
    Tps tps;     // TPS for prompt, generate, etc

    KPIs() { reset(); }
    void reset();  // reset to initial state

    QUALLA_API std::string dump(
        std::string_view sep = " ") const;  // dump KPIs as a formated string
  };

  // Get latest KPIs.
  // Updates TPS, etc as needed.
  QUALLA_API virtual KPIs& kpis();
  Engine& engine() { return *_engine; }

 protected:
  const std::string _type;
  KPIs _kpis;
  std::shared_ptr<Env> _env;        // Shared between multiple Decoder and Encoder
  std::shared_ptr<Engine> _engine;  // engines
};

}  // namespace qualla

#endif  // QUALLA_DIALOG_HPP
