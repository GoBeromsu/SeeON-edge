// Synthetic pthread fixture for seeon_media_stop's frozen half-budget.
//
// The control thread is test-owned and empty: no pipeline, source element, or
// Smart Record object. This observes the API stop split and a real join. It
// does not measure vendor NULL latency and must not emit start-sr or stop-sr.

#include "media_internal.h"
#include <chrono>
#include <cstdio>
#include <thread>

using namespace seeon_media;

namespace {

int failures = 0;

#define CHECK(condition) do { \
  if (!(condition)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    ++failures; \
  } \
} while (0)

struct StopProbe {
  std::unique_ptr<SeeonMedia> media = std::make_unique<SeeonMedia>();
  std::atomic<bool> entered{false};
  std::atomic<bool> release{false};
  bool created = false;
  ~StopProbe() {
    release.store(true);
    if (created && !media->control_joined) pthread_join(media->control, nullptr);
  }
};

void *stop_probe_thread(void *data) {
  auto *probe = static_cast<StopProbe *>(data);
  probe->entered.store(true);
  while (!probe->release.load())
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  probe->media->exited.store(true);
  return nullptr;
}

void stop_freezes_teardown_without_vendor_latency_claim(uint32_t deadline_ms) {
  StopProbe probe;
  auto &media = *probe.media;
  if (pthread_create(&media.control, nullptr, stop_probe_thread, &probe) != 0) {
    std::fprintf(stderr, "FAIL pthread_create for deadline %u\n", deadline_ms);
    ++failures;
    return;
  }
  probe.created = true;
  const auto entered_deadline = Clock::now() + std::chrono::seconds(2);
  while (!probe.entered.load() && Clock::now() < entered_deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  if (!probe.entered.load()) {
    std::fprintf(stderr, "FAIL stop-probe thread did not enter for deadline %u\n", deadline_ms);
    ++failures;
    return;
  }

  const auto before = Clock::now();
  const auto first = seeon_media_stop(probe.media.get(), deadline_ms);
  const auto after = Clock::now();
  const auto budget = std::chrono::milliseconds(finalize_budget_ms(deadline_ms));
  CHECK(media.stop_requested.load());
  CHECK(!media.admitting.load());
  CHECK(media.teardown_budget_set);
  CHECK(media.teardown_requested.load());
  const auto frozen = media.teardown_at;
  CHECK(frozen >= before + budget);
  CHECK(frozen <= after + budget);
  CHECK(first == SEEON_MEDIA_FATAL);
  CHECK(unpack(media.fatal.load()).code == SEEON_MEDIA_ERROR_STOP_TIMEOUT);
  CHECK(media.stop_timed_out.load());

  // api() does not return early on an existing fatal. Release before the
  // repeated stop so its fresh deadline can observe the actual join.
  probe.release.store(true);
  const auto again = seeon_media_stop(probe.media.get(), 2000);
  CHECK(again == SEEON_MEDIA_OK);
  CHECK(media.control_joined);
  CHECK(media.exited.load());
  CHECK(media.teardown_at == frozen);
  CHECK(media.teardown_budget_set);
  CHECK(media.warnings.load() == 0);
  CHECK(unpack(media.fatal.load()).code == SEEON_MEDIA_ERROR_STOP_TIMEOUT);
}

} // namespace

int main() {
  std::fprintf(stderr,
      "seeon media shutdown tests: empty-model pthread fixture; "
      "not vendor NULL or Smart Record latency\n");
  CHECK(finalize_budget_ms(0) == 0);
  CHECK(finalize_budget_ms(1) == 0);
  stop_freezes_teardown_without_vendor_latency_claim(0);
  stop_freezes_teardown_without_vendor_latency_claim(1);
  if (failures) {
    std::fprintf(stderr, "%d shutdown assertion(s) failed\n", failures);
    return 1;
  }
  std::fprintf(stderr, "shutdown pthread fixture passed\n");
  return 0;
}
