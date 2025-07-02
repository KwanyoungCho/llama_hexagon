//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================
#ifndef QUALLA_DETAIL_BUFFER_HPP
#define QUALLA_DETAIL_BUFFER_HPP

namespace qualla {
class Buffer {
 public:
  Buffer(uint32_t buffsize) {
    m_buff          = (uint8_t *)malloc(buffsize);
    m_buffSize      = buffsize;
    m_position      = 0;
    m_position_read = 0;
  }

  ~Buffer() {
    if (m_buff != nullptr) free(m_buff);
  }

  uint8_t *getBuffer() { return m_buff; }

  uint8_t *getBufferRef(const uint32_t size) { return (m_buff + size); }

  uint32_t getBufferSize() { return m_buffSize; }

  void appendBuffer(uint8_t *buff, uint32_t size) {
    memcpy(m_buff + m_position, buff, size);
    m_position = m_position + size;
  }

  void incrementalCopy(uint8_t *dest, uint32_t size) {
    memcpy(dest, m_buff + m_position_read, size);
    m_position_read += size;
  }

  void setPosFromCurr(int32_t rel_pos_from_curr) { m_position += rel_pos_from_curr; }

  void reset() {
    if (m_buff) free(m_buff);

    m_position      = 0;
    m_position_read = 0;
  }

 private:
  uint8_t *m_buff;
  uint64_t m_buffSize{0};
  uint64_t m_position{0};
  uint64_t m_position_read{0};
};
}  // namespace qualla

#endif  // QUALLA_DETAIL_BUFFER_HPP