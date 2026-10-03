// Synthetic CPU/state-transition fixtures for native recording publication.
//
// These tests include media_recording.cpp in this translation unit so private
// helpers are observable without a production seam. They do not compile or
// link that file a second time. They never emit start-sr or stop-sr, never
// invent a vendor filepath, and never claim Smart Record, GPU, or fsync
// acceptance. A passing run proves only the modeled transitions below.
// Real pthread teardown is in media_shutdown_tests.cpp.

#include "media_internal.h"
// Test-only inclusion. The Make target must not also compile this file.
#include "media_recording.cpp"
#include <chrono>
#include <cstdio>

using namespace seeon_media;

namespace {

int failures = 0;

#define CHECK(condition) do { \
  if (!(condition)) { \
    std::fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #condition); \
    ++failures; \
  } \
} while (0)

SeeonMediaRecord read_record(SeeonMedia &media) {
  SeeonMediaRecord record{};
  const auto status = seeon_media_try_read_record(&media, &record, sizeof(record));
  CHECK(status == SEEON_MEDIA_OK);
  return record;
}

bool record_empty(const SeeonMediaRecord &record) {
  return record.duration_ms == 0 && record.width == 0 && record.height == 0 &&
      record.contains_video == 0 && record.contains_audio == 0 &&
      record.directory[0] == '\0' && record.filename[0] == '\0';
}

struct Fixture {
  // Heap-owned: SeeonMedia carries the full source/preview tables.
  std::unique_ptr<SeeonMedia> media = std::make_unique<SeeonMedia>();
  static constexpr uint32_t kCapacity = 1;
  Fixture() {
    // open() allocates this array from record_capacity before any slot use.
    media->records.reset(new RecordSlot[kCapacity]);
    media->config.record_capacity = kCapacity;
    media->config.record_cache_seconds = 10;
    media->config.source_count = 1;
    auto &source = media->sources[0];
    source.index = 0;
    source.rtsp = true;
    source.binding = {7, 1, 1};
    source.linked.store(true);
    source.frames.store(1);
    source.active_record.store(-1);
    // Null on purpose: a test that reaches start-sr or stop-sr must fault.
    source.element = nullptr;
    media->state.store(SEEON_MEDIA_RUNNING);
    media->admitting.store(true);
  }
  SeeonMedia &operator*() { return *media; }
  SeeonMedia *operator->() { return media.get(); }
  SeeonMedia *get() { return media.get(); }
  RecordSlot &occupy(RecordPhase phase, uint64_t request, uint64_t session, bool pending) {
    auto &slot = media->records[0];
    auto &registration = media->lifetime->records[0];
    registration.owner = media.get();
    registration.source = &media->sources[0];
    registration.slot = &slot;
    registration.request = request;
    registration.token.store(request);
    registration.entries.state.store(0);
    slot.owner = media.get();
    slot.source = &media->sources[0];
    slot.registration = &registration;
    slot.token = request;
    slot.ticket = {media->sources[0].binding, request, 0, 0, 0, 0};
    slot.session.store(session);
    slot.sdk_pending = pending;
    slot.action_running = false;
    slot.stop_sent = false;
    slot.callback_processed = false;
    slot.stop_requested.store(false);
    slot.callback_claimed.store(false);
    slot.callback_done.store(false);
    slot.phase.store(phase);
    media->sources[0].active_record.store(0);
    media->records_used.store(1);
    return slot;
  }
};

void queued_global_stop_is_cancelled_without_sdk_action() {
  Fixture fixture;
  auto &media = *fixture;
  SeeonMediaRecordTicket ticket{};
  const SeeonMediaBinding binding = media.sources[0].binding;
  CHECK(seeon_media_record_start(fixture.get(), 0, &binding, 1, 1, 1, &ticket, sizeof(ticket)) ==
      SEEON_MEDIA_OK);
  CHECK(ticket.session_valid == 0);
  CHECK(media.records[0].phase.load() == RecordPhase::Queued);
  CHECK(media.records[0].registration == &media.lifetime->records[0]);
  CHECK(media.sources[0].active_record.load() == 0);
  CHECK(!media.records[0].sdk_pending);
  CHECK(!media.records[0].action_running);
  CHECK(!media.records[0].stop_sent);

  media.stop_requested.store(true);
  const auto record = read_record(media);
  CHECK(record.result == SEEON_MEDIA_STALE);
  CHECK(record.error == SEEON_MEDIA_ERROR_CANCELLED);
  CHECK(record_empty(record));
  CHECK(record.ticket.request_id == 1);
  CHECK(!media.records[0].sdk_pending);
  CHECK(!media.records[0].stop_requested.load());
  CHECK(!media.records[0].stop_sent);
  CHECK(media.warnings.load() == 0);
  CHECK(media.fatal.load() == 0);
}

void started_before_deadline_stays_active_until_global_stop_requests_stop() {
  Fixture fixture;
  auto &slot = fixture.occupy(RecordPhase::Active, 4, 9, true);
  slot.deadline = Clock::now() + std::chrono::hours(1);
  auto &media = *fixture;

  SeeonMediaRecord unread{};
  CHECK(seeon_media_try_read_record(fixture.get(), &unread, sizeof(unread)) == SEEON_MEDIA_EMPTY);
  CHECK(slot.phase.load() == RecordPhase::Active);
  CHECK(!slot.stop_requested.load());
  CHECK(!slot.stop_sent);
  CHECK(slot.sdk_pending);
  CHECK(slot.result.directory[0] == '\0');

  media.stop_requested.store(true);
  CHECK(seeon_media_try_read_record(fixture.get(), &unread, sizeof(unread)) == SEEON_MEDIA_EMPTY);
  CHECK(slot.phase.load() == RecordPhase::Active);
  CHECK(slot.stop_requested.load());
  CHECK(!slot.stop_sent);
  CHECK(slot.sdk_pending);
  CHECK(slot.result.filename[0] == '\0');
  CHECK(slot.result.duration_ms == 0);
  CHECK(media.warnings.load() == 0);
}

void fatal_wins_over_queued_slot() {
  Fixture fixture;
  auto &queued = fixture.occupy(RecordPhase::Queued, 5, kNoSession, false);
  queued.deadline = Clock::now() + std::chrono::hours(1);
  fixture->fail(SEEON_MEDIA_ERROR_STATE);

  const auto record = read_record(*fixture);
  CHECK(record.result == SEEON_MEDIA_FATAL);
  CHECK(record.error == SEEON_MEDIA_ERROR_STATE);
  CHECK(record_empty(record));
  CHECK(!queued.sdk_pending);
  CHECK(!queued.stop_requested.load());
  CHECK(fixture->warnings.load() == 0);
}

void fatal_wins_over_started_slot() {
  Fixture fixture;
  auto &started = fixture.occupy(RecordPhase::Active, 6, 3, true);
  started.deadline = Clock::now() + std::chrono::hours(1);
  fixture->fail(SEEON_MEDIA_ERROR_BUS);

  const auto record = read_record(*fixture);
  CHECK(record.result == SEEON_MEDIA_FATAL);
  CHECK(record.error == SEEON_MEDIA_ERROR_BUS);
  CHECK(record_empty(record));
  CHECK(started.stop_requested.load());
  CHECK(started.sdk_pending);
  CHECK(fixture->warnings.load() == 0);
}

void slot_deadline_expires_truthfully() {
  Fixture fixture;
  auto &slot = fixture.occupy(RecordPhase::Queued, 7, kNoSession, false);
  slot.deadline = Clock::now() - std::chrono::milliseconds(1);

  const auto record = read_record(*fixture);
  CHECK(record.result == SEEON_MEDIA_FATAL);
  CHECK(record.error == SEEON_MEDIA_ERROR_RECORD_TIMEOUT);
  CHECK(record_empty(record));
  CHECK(fixture->warnings.load() == 1);
  CHECK(unpack(fixture->warning.load()).code == SEEON_MEDIA_ERROR_RECORD_TIMEOUT);
  CHECK(!slot.sdk_pending);
}

void pending_started_recording_survives_until_modeled_null_cancellation() {
  Fixture fixture;
  auto &slot = fixture.occupy(RecordPhase::Active, 8, 11, true);
  slot.deadline = Clock::now() + std::chrono::hours(1);
  auto &media = *fixture;
  media.stop_requested.store(true);

  SeeonMediaRecord unread{};
  CHECK(seeon_media_try_read_record(fixture.get(), &unread, sizeof(unread)) == SEEON_MEDIA_EMPTY);
  CHECK(recording_pending(media));
  CHECK(slot.phase.load() == RecordPhase::Active);
  CHECK(slot.stop_requested.load());
  CHECK(slot.sdk_pending);
  CHECK(media.warnings.load() == 0);

  // Models the post-NULL control path. This call is not itself an SDK NULL.
  recording_cancel(media);
  CHECK(!recording_pending(media));
  const auto record = read_record(media);
  CHECK(record.result == SEEON_MEDIA_STALE);
  CHECK(record.error == SEEON_MEDIA_ERROR_CANCELLED);
  CHECK(record_empty(record));
  CHECK(media.warnings.load() == 0);
}

void missing_callback_after_stop_sent_warns_without_invented_media() {
  Fixture fixture;
  auto &slot = fixture.occupy(RecordPhase::Active, 9, 12, true);
  slot.stop_sent = true;

  recording_cancel(*fixture);
  const auto record = read_record(*fixture);
  CHECK(record.result == SEEON_MEDIA_STALE);
  CHECK(record.error == SEEON_MEDIA_ERROR_CANCELLED);
  CHECK(record_empty(record));
  CHECK(fixture->warnings.load() == 1);
  CHECK(unpack(fixture->warning.load()).code == SEEON_MEDIA_ERROR_RECORD_TIMEOUT);
  CHECK(!slot.sdk_pending);
  CHECK(slot.result.directory[0] == '\0');
}

void repeated_cancellation_preserves_delivered_result_and_warning() {
  Fixture fixture;
  auto &slot = fixture.occupy(RecordPhase::Active, 10, 13, true);
  slot.stop_sent = true;
  auto &media = *fixture;

  recording_cancel(media);
  CHECK(media.warnings.load() == 1);
  CHECK(slot.phase.load() == RecordPhase::Ready);
  CHECK(media.records_used.load() == 1);

  recording_cancel(media);
  CHECK(slot.result.result == SEEON_MEDIA_STALE);
  CHECK(slot.result.error == SEEON_MEDIA_ERROR_CANCELLED);
  CHECK(record_empty(slot.result));
  CHECK(media.warnings.load() == 1);
  CHECK(slot.phase.load() == RecordPhase::Ready);
  const auto delivered = read_record(media);
  CHECK(delivered.result == SEEON_MEDIA_STALE);
  CHECK(delivered.error == SEEON_MEDIA_ERROR_CANCELLED);
  CHECK(record_empty(delivered));
  CHECK(slot.phase.load() == RecordPhase::Unused);
  CHECK(media.records_used.load() == 0);
  CHECK(media.sources[0].active_record.load() == -1);
}

void finalize_split_uses_floor_half() {
  CHECK(finalize_budget_ms(0) == 0);
  CHECK(finalize_budget_ms(1) == 0);
  CHECK(finalize_budget_ms(5) == 2);
  CHECK(finalize_budget_ms(UINT32_MAX) == UINT32_MAX / 2);
}

} // namespace

int main() {
  std::fprintf(stderr,
      "seeon media recording tests: synthetic state fixtures only; "
      "not Smart Record, GPU, filepath, or fsync evidence\n");
  queued_global_stop_is_cancelled_without_sdk_action();
  started_before_deadline_stays_active_until_global_stop_requests_stop();
  fatal_wins_over_queued_slot();
  fatal_wins_over_started_slot();
  slot_deadline_expires_truthfully();
  pending_started_recording_survives_until_modeled_null_cancellation();
  missing_callback_after_stop_sent_warns_without_invented_media();
  repeated_cancellation_preserves_delivered_result_and_warning();
  finalize_split_uses_floor_half();
  if (failures) {
    std::fprintf(stderr, "%d recording state assertion(s) failed\n", failures);
    return 1;
  }
  std::fprintf(stderr, "recording state fixtures passed\n");
  return 0;
}
