#ifndef SEEON_MEDIA_INTERNAL_H
#define SEEON_MEDIA_INTERNAL_H

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include "media_runtime.h"
#include <pthread.h>
#include <gst/app/gstappsink.h>
#include <gst/gst.h>
#include "gst-nvdssr.h"
#include "gstnvdsmeta.h"
#include "gstnvdsinfer.h"
#include <array>
#include <atomic>
#include <chrono>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace seeon_media {
using Clock = std::chrono::steady_clock;
constexpr uint32_t kMaxUserMeta = 64, kMaxLayers = 32;
constexpr uint32_t kProbeFailureThreshold = 3;
constexpr uint64_t kNoSession = UINT64_MAX;
// First stop call owns this split. Later calls must not move it: a repeated
// stop with a longer deadline cannot reclaim time already reserved for NULL.
constexpr uint32_t finalize_budget_ms(uint32_t deadline_ms) noexcept {
  return deadline_ms / 2;
}
static_assert(std::atomic<uint64_t>::is_always_lock_free, "SDK counters must not block");
struct Failure { SeeonMediaError code; };
inline void require(bool ok, SeeonMediaError code) { if (!ok) throw Failure{code}; }
inline bool same_binding(const SeeonMediaBinding &a, const SeeonMediaBinding &b) noexcept {
  return a.token == b.token && a.generation == b.generation && a.epoch == b.epoch;
}
inline uint64_t diagnostic(SeeonMediaError code, uint32_t severity,
                           uint32_t domain = 0, int32_t sdk_code = 0) noexcept {
  return uint64_t(code) | (uint64_t(domain) << 16) | (uint64_t(severity) << 24) |
         (uint64_t(uint32_t(sdk_code)) << 32);
}
inline SeeonMediaDiagnostic unpack(uint64_t value) noexcept {
  return {uint32_t((value >> 24) & 255), uint32_t(value & 65535),
          uint32_t((value >> 16) & 255), int32_t(value >> 32)};
}

// Retirement and entry share one atomic modification order. A rejected entry
// never increments the count, so reopening a drained record registration cannot
// race a delayed decrement from a rejected callback.
struct LeaseGate {
  static constexpr uint64_t retired = uint64_t(1) << 63;
  std::atomic<uint64_t> state;
  explicit LeaseGate(bool closed = false) noexcept : state(closed ? retired : 0) {}
  bool enter() noexcept {
    auto value = state.load();
    while (!(value & retired)) {
      if (value == retired - 1) return false;
      if (state.compare_exchange_weak(value, value + 1)) return true;
    }
    return false;
  }
  void leave() noexcept { state.fetch_sub(1); }
  void retire() noexcept { state.fetch_or(retired); }
  bool drained() const noexcept { return (state.load() & ~retired) == 0; }
};
struct Source;
struct RecordSlot;
struct RecordRegistration {
  LeaseGate entries{true};
  std::atomic<uintptr_t> token{0};
  // Immutable between publication and retirement + the last lease release.
  SeeonMedia *owner = nullptr;
  Source *source = nullptr;
  RecordSlot *slot = nullptr;
  uint64_t request = 0;
};
struct CallbackLifetime {
  LeaseGate entries;
  std::atomic<uint32_t> registrations{0};
  std::array<RecordRegistration, SEEON_MEDIA_MAX_RECORDS> records;
};
struct CallbackRegistration {
  std::shared_ptr<CallbackLifetime> lifetime;
  SeeonMedia *owner = nullptr;
  uint32_t source = SEEON_MEDIA_MAX_SOURCES;
  LeaseGate entries;
  std::atomic<bool> detached{false};
};
// The SDK owns this box until its GClosure/GstPadProbe/bus destroy notifier.
// An invocation's SDK reference protects the box even before our first opcode.
using CallbackData = std::shared_ptr<CallbackRegistration>;

struct Source {
  uint32_t index = 0;
  SeeonMediaBinding binding{};
  std::string uri, prefix;
  bool rtsp = false;
  GstElement *element = nullptr;
  GstPad *mux_pad = nullptr;
  std::atomic<GstPad *> video_pad{nullptr}; // One owned reference, retained through callback drain.
  gulong video_probe = 0; // Only the pad claim winner writes; teardown reads after callback drain.
  gulong pad_signal = 0, removed_signal = 0, record_signal = 0;
  std::atomic<bool> linked{false};
  std::atomic<bool> negotiated{false};
  std::atomic<int> active_record{-1};
  std::atomic<uint64_t> frames{0}, overwritten{0}, dropped{0}, malformed{0};
  std::atomic<uint64_t> tensor_absent{0}, objects{0};
  uint32_t consecutive_failures = 0; // Single tracker streaming task.
  std::mutex pose_mutex;
  bool pose_ready = false;
  SeeonMediaPose pose{};
  SeeonMediaPose pose_staging{}; // Bounded validation storage; never published partially.
};

enum class RecordPhase { Unused, Queued, Starting, Active, Ready, Consumed };
struct RecordSlot {
  SeeonMedia *owner = nullptr;
  Source *source = nullptr;
  SeeonMediaRecordTicket ticket{}; // Immutable after Queued is published.
  uint32_t lookback = 0, forward = 0;
  Clock::time_point deadline{};
  std::atomic<RecordPhase> phase{RecordPhase::Unused};
  std::atomic<uint64_t> session{kNoSession};
  std::atomic<bool> stop_requested{false}, callback_claimed{false}, callback_done{false};
  RecordRegistration *registration = nullptr;
  uintptr_t token = 0;
  // Control/API state below is protected by records_mutex, not callback entry.
  bool sdk_pending = false, action_running = false;
  bool stop_sent = false, callback_processed = false;
  SeeonMediaRecord callback_result{}; // SDK writer; callback_done publishes it.
  Clock::time_point callback_completed{}; // Published with callback_result, never poll time.
  SeeonMediaRecord result{}; // Control/API writer under records_mutex; immutable once Ready.
};

enum class PreviewPhase { Idle, Queued, Draining, Armed, Inflight, Ready };
struct PreviewSampleMailbox {
  enum class Phase { Empty, Writing, Ready, Reading };
  static_assert(std::atomic<Phase>::is_always_lock_free, "Sample handoff must not block");
  std::atomic<Phase> phase{Phase::Empty};
  GstSample *sample = nullptr; // One owned reference, published together with its arrival time.
  Clock::time_point completed{};
  bool publish(GstSample *value, Clock::time_point at) noexcept {
    auto expected = Phase::Empty;
    if (!phase.compare_exchange_strong(expected, Phase::Writing)) return false;
    sample = value; completed = at;
    phase.store(Phase::Ready);
    return true;
  }
  GstSample *take(Clock::time_point *at = nullptr) noexcept {
    auto expected = Phase::Ready;
    if (!phase.compare_exchange_strong(expected, Phase::Reading)) return nullptr;
    auto *value = sample;
    if (at) *at = completed;
    sample = nullptr;
    phase.store(Phase::Empty);
    return value;
  }
};
struct Preview {
  std::mutex mutex;
  std::atomic<PreviewPhase> phase{PreviewPhase::Idle};
  std::atomic<bool> control_busy{false}, gate_idle{false};
  std::atomic<uint32_t> callbacks{0};
  PreviewSampleMailbox sample;
  bool encoder_pending = false; // Protected by mutex; survives failure consumption.
  SeeonMediaPreview result{};
  Clock::time_point deadline{};
  uint64_t last_request = 0;
  uint64_t publication_floor = 0; // Source.frames at request acceptance, never SDK frame_num.
  uint32_t source = 0;
  bool draw = false;
  std::vector<uint8_t> bytes;
  GstElement *queue = nullptr, *valve = nullptr, *bin = nullptr;
  GstElement *tiler = nullptr, *osd = nullptr, *sink = nullptr;
  GstPad *gate_pad = nullptr, *input_pad = nullptr;
  gulong gate_probe = 0, input_probe = 0, sample_signal = 0;
  CallbackData gate_registration, sample_registration;
};
} // namespace seeon_media

struct SeeonMedia {
  SeeonMediaConfig config{}; // Pointer members are cleared after copying.
  std::string infer_path, tracker_config, tracker_library, record_directory;
  std::array<seeon_media::Source, SEEON_MEDIA_MAX_SOURCES> sources;
  std::unique_ptr<seeon_media::RecordSlot[]> records;
  std::mutex records_mutex;
  std::atomic<uint32_t> records_used{0};
  uint64_t last_record_request = 0; // Serialized C caller only.
  std::shared_ptr<seeon_media::CallbackLifetime> lifetime =
      std::make_shared<seeon_media::CallbackLifetime>();
  seeon_media::Preview preview;
  GstElement *pipeline = nullptr, *mux = nullptr, *tee = nullptr;
  GstBus *bus = nullptr;
  GstPad *metadata_pad = nullptr;
  gulong metadata_probe = 0;
  std::array<GstPad *, 2> tee_pads{};
  pthread_t control{};
  bool control_joined = false; // Serialized C owner only; Linux DeepStream.
  std::atomic<SeeonMediaState> state{SEEON_MEDIA_OPEN};
  std::atomic<bool> admitting{false}, stop_requested{false}, exited{false};
  // API stop sets this once, at half the first deadline. Control reads it.
  // Recording finalize runs after stop_requested and before this flag.
  std::atomic<bool> teardown_requested{false};
  seeon_media::Clock::time_point teardown_at{};
  bool teardown_budget_set = false; // API owner only. First stop freezes teardown_at.
  std::atomic<bool> stop_timed_out{false};
  std::atomic<uint64_t> fatal{0}, warning{0}, warnings{0}, capacity_refusals{0};
  std::atomic<uint64_t> preview_dropped{0}, late_record_callbacks{0};

  void fail(SeeonMediaError code, uint32_t domain = 0, int32_t sdk_code = 0) noexcept {
    uint64_t empty = 0;
    fatal.compare_exchange_strong(empty, seeon_media::diagnostic(code, 2, domain, sdk_code));
    admitting.store(false);
  }
  void warn(SeeonMediaError code, uint32_t domain = 0, int32_t sdk_code = 0) noexcept {
    warning.store(seeon_media::diagnostic(code, 1, domain, sdk_code));
    warnings.fetch_add(1);
  }
  SeeonMediaResult capacity() noexcept {
    capacity_refusals.fetch_add(1);
    warn(SEEON_MEDIA_ERROR_CAPACITY);
    return SEEON_MEDIA_BUSY;
  }
};

namespace seeon_media {
struct CallbackGuard {
  CallbackRegistration *registration = nullptr;
  explicit CallbackGuard(gpointer data) noexcept {
    auto *context = static_cast<CallbackData *>(data)->get();
    if (!context->lifetime->entries.enter()) return;
    if (!context->entries.enter()) {
      context->lifetime->entries.leave();
      return;
    }
    registration = context;
  }
  explicit operator bool() const noexcept { return registration != nullptr; }
  SeeonMedia &media() const noexcept { return *registration->owner; }
  uint32_t source_index() const noexcept { return registration->source; }
  ~CallbackGuard() {
    if (!registration) return;
    registration->entries.leave();
    registration->lifetime->entries.leave();
  }
  CallbackGuard(const CallbackGuard &) = delete;
  CallbackGuard &operator=(const CallbackGuard &) = delete;
};
struct RecordLease {
  RecordRegistration *registration = nullptr;
  RecordLease(CallbackLifetime &lifetime, uintptr_t token) noexcept {
    if (!token) return;
    for (auto &entry : lifetime.records) {
      if (entry.token.load() != token || !entry.entries.enter()) continue;
      // A lookup paused before entry can resume after reuse. Rechecking under
      // the lease rejects that old token before reading any mutable mapping.
      if (entry.token.load() == token) { registration = &entry; return; }
      entry.entries.leave();
    }
  }
  ~RecordLease() { if (registration) registration->entries.leave(); }
  RecordLease(const RecordLease &) = delete;
  RecordLease &operator=(const RecordLease &) = delete;
};
struct PreviewGuard {
  Preview &preview;
  explicit PreviewGuard(Preview &p) noexcept : preview(p) { preview.callbacks.fetch_add(1); }
  ~PreviewGuard() { preview.callbacks.fetch_sub(1); }
};
template<class F> SeeonMediaResult api(SeeonMedia *m, F &&body) noexcept {
  if (!m) return SEEON_MEDIA_FATAL;
  try { return body(); }
  catch (const Failure &failure) { m->fail(failure.code); }
  catch (...) { m->fail(SEEON_MEDIA_ERROR_EXCEPTION); }
  return SEEON_MEDIA_FATAL;
}
inline SeeonMediaResult source_check(SeeonMedia &m, uint32_t id,
                                     const SeeonMediaBinding *binding) noexcept {
  if (id >= m.config.source_count || !binding) return SEEON_MEDIA_STALE;
  return same_binding(m.sources[id].binding, *binding) ? SEEON_MEDIA_OK : SEEON_MEDIA_STALE;
}
inline SeeonMediaResult admission(SeeonMedia &m) noexcept {
  if (m.fatal.load()) return SEEON_MEDIA_FATAL;
  return !m.stop_requested.load() && m.admitting.load() && m.state.load() == SEEON_MEDIA_RUNNING ?
      SEEON_MEDIA_OK : SEEON_MEDIA_BUSY;
}
inline SeeonMediaRecordTicket ticket(const RecordSlot &slot, bool coalesced = false) noexcept {
  auto result = slot.ticket;
  const auto session = slot.session.load();
  result.session_valid = session != kNoSession;
  result.session_id = result.session_valid ? uint32_t(session) : 0;
  result.coalesced = coalesced;
  return result;
}

NvDsMetaType publication_meta_type() noexcept; // Registered before installing the metadata probe.
GstPadProbeReturn metadata_probe(GstPad *, GstPadProbeInfo *, gpointer) noexcept;
GstPadProbeReturn preview_input(GstPad *, GstPadProbeInfo *, gpointer) noexcept;
GstPadProbeReturn preview_idle(GstPad *, GstPadProbeInfo *, gpointer) noexcept;
GstFlowReturn preview_sample(GstAppSink *, gpointer) noexcept;
gulong connect_callback(SeeonMedia &, GstElement *, const char *, GCallback,
    uint32_t source = SEEON_MEDIA_MAX_SOURCES, CallbackData *registration = nullptr);
gulong add_probe(SeeonMedia &, GstPad *, GstPadProbeType, GstPadProbeCallback,
    uint32_t source = SEEON_MEDIA_MAX_SOURCES, CallbackData *registration = nullptr);
void preview_tick(SeeonMedia &);
void preview_expire(SeeonMedia &); // Requires preview.mutex.
void preview_cancel(SeeonMedia &); // Control task, after callbacks disconnect.
void record_done(GstElement *, gpointer, gpointer, gpointer) noexcept;
void recording_tick(SeeonMedia &);
void recording_cancel(SeeonMedia &);
// True while a started session can still receive a genuine sr-done receipt.
bool recording_pending(SeeonMedia &);
void disconnect(SeeonMedia &) noexcept;
void release_graph(SeeonMedia &) noexcept; // Only before start or after quiescence.
} // namespace seeon_media
#endif
