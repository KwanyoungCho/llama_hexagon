//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_DETAIL_KPI_HPP
#define QUALLA_DETAIL_KPI_HPP

#include <chrono>
#include <cstdint>
#include <string>
#include <string_view>

namespace qualla {

struct Kpi {
  uint64_t count{0ul};       // number of events
  uint64_t last_usec{0ul};   // usec spent on the last event
  uint64_t total_usec{0ul};  // total usec spent on this event
  uint64_t min_usec{0ul};    // min usec spent on any event
  uint64_t max_usec{0ul};    // max usec spend on any event

  std::string dump(std::string_view sep = " ") const;

  void reset() {
    count      = 0;
    total_usec = 0;
    last_usec  = 0;
    min_usec   = ~0UL;
    max_usec   = 0;
  }

  void update(uint64_t usec) {
    ++count;
    last_usec = usec;
    total_usec += usec;
    if (usec > max_usec) max_usec = usec;
    if (usec < min_usec) min_usec = usec;
  }
};

}  // namespace qualla

#endif  // QUALLA_DETAIL_KPI_HPP
