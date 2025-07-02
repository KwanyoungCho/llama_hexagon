//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

// Based on Tokenizer.cpp from MLC-LLM project.
// Copyright (c) 2023 by Contributors

#ifndef QUALLA_TOKENIZER_HPP
#define QUALLA_TOKENIZER_HPP

#include <filesystem>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include "qualla/context.hpp"
#include "qualla/detail/exports.h"

namespace qualla {

/*!
 * \brief a universal tokenizer that loads
 *  either HF's tokenizer or sentence piece,
 *  depending on the constructor
 */
class Tokenizer : public State {
 public:
  /*! \brief virtual destructor */
  virtual ~Tokenizer() {}

  /*! \brief clean Up dangling history*/
  QUALLA_API virtual void cleanUp() = 0;

  /*!
   * \brief Encode text into ids.
   * \param text The input text.
   * \returns The encoded token ids.
   */
  QUALLA_API virtual std::vector<int32_t> encode(const std::string& text) = 0;

  // Encode text directly into token vector appending to existing tokens.
  // Return number of appended tokens.
  QUALLA_API virtual size_t encode(const std::string& text, std::vector<int32_t>& tokens) = 0;

  // Use an additional flag add_bos to decide whether to add BOS token or not.
  QUALLA_API virtual size_t encode(const std::string& text,
                                   std::vector<int32_t>& tokens,
                                   bool add_bos) = 0;

  /*!
   * \brief Decode token ids into text.
   * \param text The token ids.
   * \returns The decoded text.
   */
  QUALLA_API virtual std::string decode(const std::vector<int32_t>& ids) = 0;

  //---------------------------------------------------
  // Factory functions from byte-blobs
  // These factory function takes in in-memory blobs
  // so the library can be independent from filesystem
  //---------------------------------------------------
  /*!
   * \brief Create HF tokenizer from a single in-memory json blob.
   *
   * \param json_blob The json blob.
   * \return The created tokenzier.
   */
  QUALLA_API static std::unique_ptr<Tokenizer> create(Context& ctx, std::istream& json_stream);
  QUALLA_API static std::unique_ptr<Tokenizer> create(Context& ctx,
                                                      const std::filesystem::path& json_path);
};

}  // namespace qualla
#endif  // QUALLA_TOKENIZER_HPP
