//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#ifndef QUALLA_DETAIL_STATE_HPP
#define QUALLA_DETAIL_STATE_HPP

#include <atomic>
#include <sstream>
#include <string>

namespace qualla {

class State {
 public:
  void clear() {
    _busy     = false;
    _canceled = false;
    _failed   = false;
    _error.clear();
  }

  void error(const std::string& e) {
    _failed = true;
    _error  = e;
  }
  void error(const std::stringstream& e) { error(e.str()); }

  void fatal(const std::string& e) {
    _fatal = true;
    error(e);
  }
  void fatal(const std::stringstream& e) { fatal(e.str()); }

  void busy(bool v = true) { _busy = v; }

  void cancel() { _canceled = true; }

  bool busy() const { return _busy; }
  bool canceled() const { return _canceled; }
  bool failed() const { return _failed || _fatal; }
  const std::string& error() const { return _error; }

 private:
  std::atomic<bool> _busy{false};      // procesing in progress
  std::atomic<bool> _canceled{false};  // cancel current processing
  std::atomic<bool> _failed{false};    // init or processing failed, can be cleared
  std::atomic<bool> _fatal{false};     // init or processing failed, cannot be cleared
  std::string _error;                  // failure reason
};

}  // namespace qualla

#endif  // QUALLA_DETAIL_STATE_HPP
