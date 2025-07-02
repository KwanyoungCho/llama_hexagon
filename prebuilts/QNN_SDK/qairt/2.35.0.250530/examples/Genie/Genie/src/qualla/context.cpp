//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include <fmt/format.h>
#include <fmt/ranges.h>

#include "qualla/context.hpp"
#include "qualla/detail/config.hpp"
#include "qualla/detail/onload.hpp"

#define __DEBUG(__fmt, ...) \
  _LOG(_env.logger(), GENIE_LOG_LEVEL_VERBOSE, fmt::format(__fmt, ##__VA_ARGS__))
namespace qualla {

Context::Context(Env& env, const std::string& name, const qualla::json& json)
    : _name(name), _env(env), _conf(json) {
  __DEBUG("ctx-new: {} config {}", _name, _conf.dump());

  qualla::Config conf(json, "context:");
  _size          = conf.optional<size_t>("size", 1024);
  _size          = conf.optional<size_t>("n-ctx", _size);  // alternative name
  _n_vocab       = conf.optional<size_t>("n-vocab", 32000);
  _draft_n_vocab = conf.optional<size_t>(
      "draft-n-vocab",
      _n_vocab);  // draft model vocab, default both target and draft have same vocab.
  _n_embd             = conf.optional<size_t>("n-embd", 1024);
  _bos_tok            = conf.optional<int32_t>("bos-token", -1);
  _embedding_length   = conf.optional<int32_t>("embedding-length", -1);
  _embedding_datatype = conf.optional<std::string>("embedding-datatype", "QNN_DATATYPE_FLOAT_32");

  // For backward compatibility. When eot-token is removed, this logic can be simplified
  // Currently, EOT is marked as default truncating token if available
  int32_t eot_tok = conf.optional<int32_t>("eot-token", -1);
  if (eot_tok >= 0) _eos_tok_list.insert(eot_tok);

  const qualla::json eos_conf = conf.optional<qualla::json>("eos-token", _eos_tok);
  if (eos_conf.is_array() && eos_conf.size() > 0) {
    const std::vector<int32_t>& eos_tokens = eos_conf.get<std::vector<int32_t>>();
    _eos_tok                               = eos_tokens[0];
    for (const int32_t& eos_tok : eos_tokens) _eos_tok_list.insert(eos_tok);
  } else if (eos_conf.is_number_integer()) {
    int32_t eos_tok = eos_conf.get<int32_t>();
    _eos_tok        = (eot_tok >= 0) ? eot_tok : eos_tok;
    _eos_tok_list.insert(eos_tok);
  }

  _pad_tok = conf.optional<qualla::json>("pad-token", _eos_tok);
}

std::unique_ptr<Context> Context::create(Env& env,
                                         const std::string& name,
                                         const qualla::json& conf) {
  return std::make_unique<Context>(env, name, conf);
}

std::unique_ptr<Context> Context::create(Env& env,
                                         const std::string& name,
                                         std::istream& json_stream) {
  return create(env, name, json::parse(json_stream));
}

std::unique_ptr<Context> Context::create(Env& env,
                                         const std::string& name,
                                         const std::string& json_str) {
  return create(env, name, json::parse(json_str));
}

#ifdef QUALLA_STATIC

// This is a hack to make sure all core bits are linked in for the static build

extern void needBasicDialog();
extern void needKvShareDialog();
extern void needSpdDialog();
extern void needSsdDialog();
extern void needLadeDialog();
extern void needMultistreamDialog();

extern void needEmbeddingEncoder();

extern void needImageEncoder();

extern void needLUTEncoder();

extern void needEagletDialog();
#ifdef QUALLA_ENGINE_QNN_HTP
extern void needQnnHtpEngine();
#endif

#ifdef QUALLA_ENGINE_QNN_CPU
extern void needQnnCpuEngine();
#endif

#ifdef QUALLA_ENGINE_QNN_GPU
extern void needQnnGpuEngine();
#endif

static OnLoad needs([]() {
  needBasicDialog();
  needKvShareDialog();
  needSpdDialog();
  needSsdDialog();
  needLadeDialog();
  needMultistreamDialog();

  needEmbeddingEncoder();
  needLUTEncoder();
  needEagletDialog();

  needImageEncoder();

#ifdef QUALLA_ENGINE_QNN_HTP
  needQnnHtpEngine();
#endif

#ifdef QUALLA_ENGINE_QNN_CPU
  needQnnCpuEngine();
#endif

#ifdef QUALLA_ENGINE_QNN_GPU
  needQnnGpuEngine();
#endif
});

#endif

}  // namespace qualla
