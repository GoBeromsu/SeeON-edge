#pragma once

// A bounded free-list pool and its RAII lease.
//
// This exists because a single TensorRT execution context serialized every
// camera in the deployment: with N sources the per-camera frame rate is
// 1/(N * critical_section), which measured 11.3fps against a 15fps target on a
// 13-camera stack. Concurrency has to be bounded and explicit - an unbounded
// pool would trade a throughput ceiling for an out-of-memory one.
//
// The pool is deliberately free of CUDA and TensorRT so its concurrency and
// exhaustion behaviour can be tested on a host without a GPU. It owns nothing:
// the caller keeps the elements alive for the pool's lifetime.

#include <condition_variable>
#include <cstddef>
#include <mutex>
#include <vector>

namespace seeon::trt {

template <typename T>
class BoundedPool {
 public:
  // Registers an element as available. Call before any acquire().
  void add(T* element) {
    std::lock_guard lock{mutex_};
    free_.push_back(element);
  }

  // Blocks until an element is free, then hands out exclusive use of it.
  T* acquire() {
    std::unique_lock lock{mutex_};
    available_.wait(lock, [this] { return !free_.empty(); });
    T* element = free_.back();
    free_.pop_back();
    return element;
  }

  void release(T* element) {
    {
      std::lock_guard lock{mutex_};
      free_.push_back(element);
    }
    available_.notify_one();
  }

  void reserve(std::size_t capacity) {
    std::lock_guard lock{mutex_};
    free_.reserve(capacity);
  }

  // Test observability only; not part of the production interface. A caller
  // must not branch on this to decide whether to acquire: the answer is stale the moment it is returned, and acquire()
  // already blocks correctly.
  [[nodiscard]] std::size_t available() {
    std::lock_guard lock{mutex_};
    return free_.size();
  }

 private:
  std::mutex mutex_;
  std::condition_variable available_;
  std::vector<T*> free_;
};

// Returns a leased element to its pool on every exit path, including the early
// returns of a failing inference. A leaked lease permanently shrinks the pool
// and slowly starves the deployment, which is precisely the kind of silent
// degradation this design is meant to make impossible.
template <typename T>
class PoolLease {
 public:
  PoolLease(BoundedPool<T>* pool, T* element) : pool_{pool}, element_{element} {}
  ~PoolLease() { pool_->release(element_); }
  PoolLease(const PoolLease&) = delete;
  PoolLease& operator=(const PoolLease&) = delete;
  PoolLease(PoolLease&&) = delete;
  PoolLease& operator=(PoolLease&&) = delete;

  [[nodiscard]] T& operator*() const { return *element_; }
  [[nodiscard]] T* operator->() const { return element_; }

 private:
  BoundedPool<T>* pool_;
  T* element_;
};

}  // namespace seeon::trt
