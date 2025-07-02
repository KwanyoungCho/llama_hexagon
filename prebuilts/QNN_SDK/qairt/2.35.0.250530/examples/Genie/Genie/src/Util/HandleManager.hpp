//==============================================================================
//
//  Copyright (c) 2019-2020 Qualcomm Technologies, Inc.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#pragma once

#include <algorithm>
#include <functional>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "HandleGenerator.hpp"

namespace qnn {
namespace util {

template <typename T>
class HandleManager {
 public:
  HandleManager()                     = default;
  HandleManager(const HandleManager&) = delete;
  HandleManager& operator=(const HandleManager&) = delete;
  HandleManager(HandleManager&&)                 = delete;
  HandleManager& operator=(HandleManager&&) = delete;

  Handle_t add(std::shared_ptr<T> item) {
    std::lock_guard<std::mutex> locker(m_itemsMtx);

    if (!item) {
      return HandleGenerator::invalid();
    }

    auto handle     = HandleGenerator::generate(item.get());
    m_items[handle] = item;
    return handle;
  }

  Handle_t add(T* item) { return add(std::shared_ptr<T>(item)); }

  Handle_t add(std::weak_ptr<T> item) { return add(item.lock()); }

  std::shared_ptr<T> get(Handle_t handle) {
    std::lock_guard<std::mutex> locker(m_itemsMtx);

    auto it = m_items.find(handle);
    if (it == m_items.end()) {
      return std::shared_ptr<T>(nullptr);
    }

    return it->second;
  }

  typedef std::function<bool(const std::pair<Handle_t, std::shared_ptr<T>>&)> UnaryPredicate_t;

  Handle_t findIf(UnaryPredicate_t pred) const {
    auto it = std::find_if(m_items.begin(), m_items.end(), pred);
    if (it == m_items.end()) {
      return HandleGenerator::invalid();
    }

    return it->first;
  }

  size_t remove(Handle_t handle) {
    std::lock_guard<std::mutex> locker(m_itemsMtx);
    return m_items.erase(handle);
  }

  void clear() { m_items.clear(); }

  const std::unordered_map<Handle_t, std::shared_ptr<T>>& getItems() const { return m_items; }

 private:
  std::unordered_map<Handle_t, std::shared_ptr<T>> m_items;
  std::mutex m_itemsMtx;
};

}  // namespace util
}  // namespace qnn
