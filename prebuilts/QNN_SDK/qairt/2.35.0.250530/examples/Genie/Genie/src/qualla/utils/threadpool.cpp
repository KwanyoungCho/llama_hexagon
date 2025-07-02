//==============================================================================
//
//  Copyright (c) Qualcomm Technologies, Inc. and/or its subsidiaries.
//  All Rights Reserved.
//  Confidential and Proprietary - Qualcomm Technologies, Inc.
//
//==============================================================================

#include "qualla/detail/threadpool.hpp"

#if defined(_WIN32)
#include "windows.h"

static bool __thread_affinity(uint64_t mask) {
  HANDLE h    = GetCurrentThread();
  DWORD_PTR m = mask;

  m = SetThreadAffinityMask(h, m);

  return m != 0;
}

static int sched_yield(void) {
  Sleep(0);
  return 0;
}

#elif defined(__APPLE__)
static bool __thread_affinity(uint64_t mask) { return true; }

#else  // posix?
#include <errno.h>
#include <sched.h>
#include <string.h>

static bool __thread_affinity(uint64_t mask) {
  cpu_set_t cpuset;
  int32_t err;

  CPU_ZERO(&cpuset);

  for (uint32_t i = 0; i < 64; i++) {
    if ((1ULL << i) & mask) {
      CPU_SET(i, &cpuset);
    }
  }

#ifdef __ANDROID__
  err = sched_setaffinity(0, sizeof(cpuset), &cpuset);
  if (err < 0) {
    err = errno;
  }
#else
  err = pthread_setaffinity_np(pthread_self(), sizeof(cpuset), &cpuset);
#endif
  if (err != 0) {
    fprintf(stderr,
            "warn: failed to set affinity mask 0x%llx (err %d: %s)\n",
            (unsigned long long)mask,
            err,
            strerror(err));
    return false;
  }

  return true;
}

#endif

#ifdef _MSC_VER

static inline void __cpu_relax(void) { YieldProcessor(); }

#else

#if defined(__aarch64__)

static inline void __cpu_relax(void) { __asm__ volatile("yield" ::: "memory"); }

#else

static inline void __cpu_relax(void) { __asm__ volatile("rep; nop" ::: "memory"); }

#endif
#endif

namespace qualla {

void ThreadPool::stop() {
  _queue_mutex.lock();
  _terminate = true;
  _queue_mutex.unlock();
  _mutex_condition.notify_all();

  for (auto& t : _threads) t.join();
  _threads.clear();
}

void ThreadPool::start(unsigned int n_threads, uint64_t cpumask, bool polling) {
  _enable_polling = polling;
  _n_threads      = n_threads ? n_threads : std::thread::hardware_concurrency();
  _cpumask        = cpumask;
  _poll           = false;  // always start non-polling (enqueue will enable as needed)
  for (uint32_t i = 0; i < _n_threads; ++i) {
    _threads.emplace_back(std::thread(&ThreadPool::loop, this, i));
  }
}

void ThreadPool::suspend() {
  std::unique_lock<std::mutex> lock(_queue_mutex);
  _poll = false;
}

void ThreadPool::loop(uint32_t ti) {
  if (_cpumask) __thread_affinity(_cpumask);

  std::unique_lock<std::mutex> lock{_queue_mutex, std::defer_lock};

  while (!_terminate) {
    lock.lock();

    if (!_jobs.empty()) {
      // Dispatch front job
      auto j = _jobs.front();
      _jobs.pop();
      lock.unlock();
      j();
    } else {
      // No jobs. Wait
      if (_poll) {
        lock.unlock();
        __cpu_relax();
      } else {
        _mutex_condition.wait(lock);
        lock.unlock();
      }
    }
  }
}

}  // namespace qualla
