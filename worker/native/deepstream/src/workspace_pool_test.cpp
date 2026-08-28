#include "workspace_pool.hpp"

#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const char* what) {
  if (!condition) {
    std::fprintf(stderr, "FAIL: %s\n", what);
    ++failures;
  }
}

struct Slot {
  int id = 0;
};

// Exclusivity is the whole point: two concurrent inferences must never bind
// tensors on the same execution context or enqueue on the same CUDA stream.
//
// Checking a peak counter alone is not enough - a peak-of-N assertion passes
// trivially when the real peak is 1, which is exactly the failure mode of a
// "pool" that never actually overlaps. So this tracks ownership by workspace
// identity (a per-slot in-use flag that must never be set twice) AND requires
// observed simultaneous ownership of every slot, forced by a barrier.
void test_acquire_is_exclusive() {
  constexpr int kPoolSize = 4;
  seeon::trt::BoundedPool<Slot> pool;
  std::vector<Slot> slots(kPoolSize);
  std::array<std::atomic<bool>, kPoolSize> in_use{};
  for (int index = 0; index < kPoolSize; ++index) {
    slots[static_cast<std::size_t>(index)].id = index;
    in_use[static_cast<std::size_t>(index)].store(false);
    pool.add(&slots[static_cast<std::size_t>(index)]);
  }

  std::atomic<bool> double_owned{false};
  std::atomic<int> concurrent{0};
  std::atomic<int> peak{0};
  std::vector<std::thread> threads;
  for (int index = 0; index < 16; ++index) {
    threads.emplace_back([&] {
      for (int iteration = 0; iteration < 64; ++iteration) {
        Slot* slot = pool.acquire();
        const seeon::trt::PoolLease<Slot> lease{&pool, slot};
        auto& flag = in_use[static_cast<std::size_t>(slot->id)];
        // A slot handed to two threads at once sets this twice.
        if (flag.exchange(true)) double_owned = true;
        const int now = concurrent.fetch_add(1) + 1;
        int previous = peak.load();
        while (now > previous && !peak.compare_exchange_weak(previous, now)) {
        }
        std::this_thread::sleep_for(std::chrono::microseconds{50});
        concurrent.fetch_sub(1);
        flag.store(false);
      }
    });
  }
  for (auto& thread : threads) thread.join();

  check(!double_owned, "no workspace was ever owned by two threads at once");
  check(peak.load() <= kPoolSize, "peak concurrency never exceeds pool size");
  // The load-bearing half: the pool must actually overlap. A serializing
  // implementation reaches peak 1 and fails here.
  check(peak.load() > 1, "inferences actually ran concurrently");
}

// Forces every workspace to be held simultaneously. If the pool only ever
// circulates one element, the barrier never releases and this deadlocks rather
// than silently passing - so the assertion below is reached only on real
// concurrency.
void test_every_workspace_is_usable_concurrently() {
  constexpr int kPoolSize = 4;
  seeon::trt::BoundedPool<Slot> pool;
  std::vector<Slot> slots(kPoolSize);
  for (int index = 0; index < kPoolSize; ++index) {
    slots[static_cast<std::size_t>(index)].id = index;
    pool.add(&slots[static_cast<std::size_t>(index)]);
  }

  std::mutex mutex;
  std::condition_variable arrived;
  int waiting = 0;
  std::set<int> held_simultaneously;
  std::vector<std::thread> threads;
  for (int index = 0; index < kPoolSize; ++index) {
    threads.emplace_back([&] {
      Slot* slot = pool.acquire();
      const seeon::trt::PoolLease<Slot> lease{&pool, slot};
      std::unique_lock lock{mutex};
      held_simultaneously.insert(slot->id);
      if (++waiting == kPoolSize) {
        arrived.notify_all();
      } else {
        arrived.wait_for(lock, std::chrono::seconds{10}, [&] { return waiting == kPoolSize; });
      }
    });
  }
  for (auto& thread : threads) thread.join();

  check(held_simultaneously.size() == static_cast<std::size_t>(kPoolSize),
        "all four workspaces were held at the same instant");
  check(pool.available() == kPoolSize, "all four returned after the barrier");
}

// A pool of one must serialize; this is the degenerate case that proves the
// wait actually blocks rather than handing out a second copy.
void test_capacity_one_serializes() {
  seeon::trt::BoundedPool<Slot> pool;
  Slot only;
  pool.add(&only);

  std::atomic<int> concurrent{0};
  std::atomic<bool> overlapped{false};
  std::vector<std::thread> threads;
  for (int index = 0; index < 8; ++index) {
    threads.emplace_back([&] {
      for (int iteration = 0; iteration < 32; ++iteration) {
        Slot* slot = pool.acquire();
        const seeon::trt::PoolLease<Slot> lease{&pool, slot};
        if (concurrent.fetch_add(1) != 0) overlapped = true;
        std::this_thread::yield();
        concurrent.fetch_sub(1);
      }
    });
  }
  for (auto& thread : threads) thread.join();

  check(!overlapped, "capacity one never overlaps");
  check(pool.available() == 1, "the single workspace came back");
}

// The inference path has several early returns for engine failures. Each one
// must hand the workspace back, or the pool shrinks by one per failure until
// the deployment deadlocks with no diagnostic.
void test_lease_returns_on_early_exit() {
  seeon::trt::BoundedPool<Slot> pool;
  std::vector<Slot> slots(2);
  for (auto& slot : slots) pool.add(&slot);

  const auto failing_inference = [&]() -> bool {
    Slot* slot = pool.acquire();
    const seeon::trt::PoolLease<Slot> lease{&pool, slot};
    return false;  // engine_enqueue_failed, engine_output_copy_failed, ...
  };
  for (int index = 0; index < 100; ++index) {
    check(!failing_inference(), "failing inference reports failure");
  }
  check(pool.available() == 2, "no capacity leaked across 100 failures");
}

// An exception unwinding out of the critical section must also return the
// workspace; the sanitized build runs this path too.
void test_lease_returns_on_throw() {
  seeon::trt::BoundedPool<Slot> pool;
  Slot only;
  pool.add(&only);

  for (int index = 0; index < 10; ++index) {
    try {
      Slot* slot = pool.acquire();
      const seeon::trt::PoolLease<Slot> lease{&pool, slot};
      throw std::runtime_error{"engine failure"};
    } catch (const std::runtime_error&) {
    }
  }
  check(pool.available() == 1, "workspace returned after unwinding");
}

// Every registered element must eventually be handed out; a pool that only
// ever issues one of its four workspaces would silently run at a quarter of
// its configured concurrency.
void test_all_elements_are_reachable() {
  seeon::trt::BoundedPool<Slot> pool;
  std::vector<Slot> slots(4);
  for (int index = 0; index < 4; ++index) {
    slots[static_cast<std::size_t>(index)].id = index;
    pool.add(&slots[static_cast<std::size_t>(index)]);
  }

  std::vector<Slot*> held;
  held.reserve(4);
  for (int index = 0; index < 4; ++index) held.push_back(pool.acquire());
  std::set<int> seen;
  for (Slot* slot : held) seen.insert(slot->id);
  check(seen.size() == 4, "all four distinct workspaces were issued");
  check(pool.available() == 0, "pool is empty when all are held");
  for (Slot* slot : held) pool.release(slot);
  check(pool.available() == 4, "pool refills");
}

}  // namespace

int main() {
  test_acquire_is_exclusive();
  test_every_workspace_is_usable_concurrently();
  test_capacity_one_serializes();
  test_lease_returns_on_early_exit();
  test_lease_returns_on_throw();
  test_all_elements_are_reachable();
  if (failures != 0) {
    std::fprintf(stderr, "%d workspace-pool check(s) failed\n", failures);
    return 1;
  }
  std::fprintf(stderr, "workspace-pool checks passed\n");
  return 0;
}
