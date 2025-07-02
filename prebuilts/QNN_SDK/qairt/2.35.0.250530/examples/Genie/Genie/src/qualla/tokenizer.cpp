//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

// Based on Tokenizers.cpp from MLC-LLM project
// Copyright (c) 2023 by Contributors

#include <fstream>

#include "qualla/tokenizer.hpp"
#include "tokenizers-capi.h"

namespace fs = std::filesystem;

namespace qualla {

/*!
 * \brief A simple c++ header of tokenizer via C API.
 */
class HFTokenizer : public Tokenizer {
 public:
  explicit HFTokenizer(Context& ctx, TokenizerHandle handle) : _ctx(ctx), _handle(handle) {}

  HFTokenizer(const HFTokenizer&) = delete;
  HFTokenizer(HFTokenizer&& other) : _ctx(other._ctx) { std::swap(other._handle, _handle); }

  ~HFTokenizer() {
    if (_handle != nullptr) {
      tokenizers_free(_handle);
    }
  }

  std::vector<int32_t> encode(const std::string& text) final {
    int add_special_token = 0;  // qualla handles special tokens at higher-level
    tokenizers_encode(_handle, text.data(), text.length(), add_special_token);
    const uint32_t* data;
    size_t len;
    tokenizers_get_encode_ids(_handle, &data, &len);
    return std::vector<int32_t>(data, data + len);
  }

  size_t encode(const std::string& text, std::vector<int32_t>& tokens) final {
    int add_special_token = 0;  // qualla handles special tokens at higher-level
    tokenizers_encode(_handle, text.data(), text.length(), add_special_token);
    const uint32_t* data;
    size_t len;
    tokenizers_get_encode_ids(_handle, &data, &len);
    tokens.reserve(tokens.size() + len);
    for (size_t i = 0; i < len; i++) tokens.push_back((int32_t)data[i]);
    return len;
  }

  size_t encode(const std::string& text, std::vector<int32_t>& tokens, bool add_bos) final {
    int add_special_token = 0;
    tokenizers_encode(_handle, text.data(), text.length(), add_special_token);
    const uint32_t* data;
    size_t len;
    tokenizers_get_encode_ids(_handle, &data, &len);
    if (add_bos) {
      if (_ctx.bos_tok() >= 0) {
        tokens.push_back(_ctx.bos_tok());
      }
    }
    tokens.reserve(tokens.size() + len);
    for (size_t i = 0; i < len; i++) tokens.push_back((int32_t)data[i]);
    return len;
  }

  std::string decode(const std::vector<int32_t>& ids) final {
    int skip_special_token = 0;

    if (!utf8_token_ids.empty()) {
      // If corrupt-UTF8 character has previously been detected,
      //      add the new token ids to the previous ids and run tokenizer.decode()
      utf8_token_ids.insert(utf8_token_ids.end(), ids.begin(), ids.end());
      tokenizers_decode(_handle,
                        reinterpret_cast<const uint32_t*>(utf8_token_ids.data()),
                        utf8_token_ids.size(),
                        skip_special_token);
    } else {
      tokenizers_decode(
          _handle, reinterpret_cast<const uint32_t*>(ids.data()), ids.size(), skip_special_token);
    }
    const char* data;
    size_t len;
    tokenizers_get_decode_str(_handle, &data, &len);
    std::string data_str = std::string(data, len);

    // Detect if the decoded string contains the corrupt-UTF8 character
    // If yes, the decode likely needs multiple tokens. Save the current token to a vector
    if (data_str.find("�") != std::string::npos) {
      // fprintf(stderr, "ERROR DETECTED");
      if (utf8_token_ids.empty())
        utf8_token_ids.insert(utf8_token_ids.end(), ids.begin(), ids.end());
      return std::string();
    }

    // If no corrupt-UTF8 character is detected, we know the token sequence produces valid UTF-8
    utf8_token_ids.clear();

    // Only handle utf-8 for
    if (ids.size() == 1 && len == 6 &&
        (data[0] == '<' && data[1] == '0' && data[2] == 'x' && data[5] == '>')) {
      // string has format "<0xNN>", where NN is the ascii code we want.
      char tmp_data[3] = {data[3], data[4], 0};
      char code        = static_cast<char>(std::strtol(tmp_data, nullptr, 16));
      // fprintf(stderr, "data=%s code=%d\n", data_str.c_str(), code);

      int firstZero = 0;
      if ((code & 1 << 7) == 0)
        firstZero = 0;
      else if ((code & 1 << 6) == 0)
        firstZero = 1;
      else if ((code & 1 << 5) == 0)
        firstZero = 2;
      else if ((code & 1 << 4) == 0)
        firstZero = 3;
      else if ((code & 1 << 3) == 0)
        firstZero = 4;
      // else throw std::runtime_error("0. Invalid utf-8 character encounterd" + data_str);
      else
        return data_str;

      // fprintf(stderr, "Code=%x (%d) firstZero=%d utf8_remaining_bytes=%d\n", code, code,
      // firstZero, utf8_remaining_bytes);
      switch (firstZero) {
        case 0:
          // This is a 1-byte UTF-8 string
          if (utf8_remaining_bytes > 0) return data_str;
          return std::string(1, code);
        case 1:
          // It is a continuation byte
          utf8_str += std::string(1, code);  // Append to buffer
          if (--utf8_remaining_bytes == 0)   // Complete utf-8 received
            return std::string(utf8_str);    // Make a copy just in case
          break;
        case 2:
        case 3:
        case 4:
          // Detected a new multi-byte utf-8 character
          utf8_str             = std::string(1, code);
          utf8_remaining_bytes = firstZero - 1;
          break;
        default:
          return data_str;
      }
      return std::string();
    }
    return data_str;
  }

  // clean the history
  void cleanUp() {
    utf8_str.clear();
    utf8_remaining_bytes = 0;
    utf8_token_ids.clear();
  }

 private:
  Context& _ctx;

  // internal handle
  TokenizerHandle _handle{nullptr};

  std::string utf8_str;
  int32_t utf8_remaining_bytes = 0;

  std::vector<int32_t> utf8_token_ids;
};

std::unique_ptr<Tokenizer> Tokenizer::create(Context& ctx, std::istream& json_stream) {
  std::string data;
  std::getline(json_stream, data, '\0');
  return std::make_unique<HFTokenizer>(ctx, tokenizers_new_from_str(data.data(), data.length()));
}

std::unique_ptr<Tokenizer> Tokenizer::create(Context& ctx, const fs::path& json_path) {
  if (!fs::exists(json_path))
    throw std::runtime_error(json_path.string() + ": file does not exist");
  std::ifstream ifs(json_path);
  return create(ctx, ifs);
}

}  // namespace qualla
