#include "media_internal.h"

using namespace seeon_media;
namespace {
// This DSO stays linked for the Worker process lifetime. Tokens are pointer-width
// opaque integers, never addresses or dereferenced pointers; no tombstone map is
// retained. Refuse exhaustion before arithmetic can wrap back to a used value.
std::atomic<uintptr_t> last_token{0};
static_assert(sizeof(uintptr_t) == sizeof(gpointer), "Record tokens must preserve pointer width");
static_assert(std::atomic<uintptr_t>::is_always_lock_free, "Token allocation must not block");
uintptr_t allocate_token() noexcept {
  auto previous = last_token.load();
  while (previous != UINTPTR_MAX) {
    if (last_token.compare_exchange_weak(previous, previous + 1)) return previous + 1;
  }
  return 0;
}
bool copy_text(char *out, const char *value) noexcept {
  if (!value) return false;
  const size_t length = strnlen(value, SEEON_MEDIA_PATH_BYTES);
  if (!length || length >= SEEON_MEDIA_PATH_BYTES) return false;
  std::memcpy(out, value, length + 1);
  return true;
}
// All helpers below run under records_mutex. Delivery is independent of SDK
// retirement: neither Ready nor Consumed releases an unresolved source operation.
void deliver(RecordSlot &slot, SeeonMediaResult result, SeeonMediaError error,
             const SeeonMediaRecord *completion = nullptr) noexcept {
  const auto phase = slot.phase.load();
  if (phase == RecordPhase::Ready || phase == RecordPhase::Consumed) return;
  slot.result = completion ? *completion : SeeonMediaRecord{};
  slot.result.ticket = ticket(slot);
  slot.result.result = result; slot.result.error = error;
  slot.phase.store(RecordPhase::Ready);
}
void reap(SeeonMedia &m, uint32_t index) noexcept {
  auto &slot = m.records[index];
  const auto phase = slot.phase.load();
  if ((phase != RecordPhase::Ready && phase != RecordPhase::Consumed) ||
      slot.sdk_pending || slot.action_running) return;
  auto &registration = *slot.registration;
  registration.entries.retire();
  if (!registration.entries.drained()) return;
  registration.token.store(0);
  int expected = int(index);
  slot.source->active_record.compare_exchange_strong(expected, -1);
  if (slot.phase.load() == RecordPhase::Consumed) {
    slot.phase.store(RecordPhase::Unused);
    m.records_used.fetch_sub(1);
  }
}
void complete_callback(SeeonMedia &m, RecordSlot &slot) noexcept {
  if (!slot.sdk_pending || slot.action_running || slot.callback_processed ||
      !slot.callback_done.load(std::memory_order_acquire)) return;
  slot.callback_processed = true;
  const auto &done = slot.callback_result;
  if (slot.session.load() == kNoSession || !done.ticket.session_valid ||
      done.ticket.session_id != slot.session.load() || done.ticket.request_id != slot.ticket.request_id ||
      !same_binding(done.ticket.binding, slot.ticket.binding) || done.ticket.source_id != slot.ticket.source_id) {
    m.fail(SEEON_MEDIA_ERROR_RECORD_IDENTITY);
    deliver(slot, SEEON_MEDIA_FATAL, SEEON_MEDIA_ERROR_RECORD_IDENTITY);
    slot.stop_requested.store(true);
    return; // Mismatched completion is not proof this SDK operation retired.
  }
  slot.sdk_pending = false;
  if (slot.phase.load() != RecordPhase::Ready && slot.phase.load() != RecordPhase::Consumed) {
    if (slot.callback_completed >= slot.deadline) {
      m.warn(SEEON_MEDIA_ERROR_RECORD_TIMEOUT);
      deliver(slot, SEEON_MEDIA_FATAL, SEEON_MEDIA_ERROR_RECORD_TIMEOUT);
    } else {
      if (done.error != SEEON_MEDIA_ERROR_NONE) m.warn(done.error);
      // A vendor filepath is completion evidence, never fsync/seal evidence.
      deliver(slot, done.result, done.error, &done);
    }
  }
}
void expire(SeeonMedia &m, RecordSlot &slot) noexcept {
  const auto phase = slot.phase.load();
  if (phase == RecordPhase::Unused || phase == RecordPhase::Ready || phase == RecordPhase::Consumed) return;
  if (m.fatal.load()) {
    deliver(slot, SEEON_MEDIA_FATAL, SeeonMediaError(unpack(m.fatal.load()).code));
  } else if (m.stop_requested.load() || (phase == RecordPhase::Queued && admission(m) != SEEON_MEDIA_OK)) {
    deliver(slot, SEEON_MEDIA_STALE, SEEON_MEDIA_ERROR_CANCELLED);
  } else if (Clock::now() >= slot.deadline) {
    m.warn(SEEON_MEDIA_ERROR_RECORD_TIMEOUT);
    deliver(slot, SEEON_MEDIA_FATAL, SEEON_MEDIA_ERROR_RECORD_TIMEOUT);
  } else {
    return;
  }
  if (slot.sdk_pending) slot.stop_requested.store(true);
}
} // namespace

namespace seeon_media {
void record_done(GstElement *element, gpointer info_pointer, gpointer request_data,
                 gpointer connected_data) noexcept {
  const auto completed = Clock::now();
  CallbackGuard guard(connected_data);
  if (!guard) return;
  // Both connected_data and request_data are admitted before owner/slot access.
  // Registry storage is independently owned by the registration lifetime domain.
  RecordLease lease(*guard.registration->lifetime, reinterpret_cast<uintptr_t>(request_data));
  auto &m = guard.media();
  if (!lease.registration) { m.late_record_callbacks.fetch_add(1); return; }
  try {
    const auto &mapping = *lease.registration;
    auto &source = m.sources[guard.source_index()];
    if (mapping.owner != &m || mapping.source != &source || element != source.element || !mapping.slot) {
      m.fail(SEEON_MEDIA_ERROR_RECORD_IDENTITY); return;
    }
    auto &slot = *mapping.slot;
    if (slot.owner != &m || slot.source != &source || mapping.request != slot.ticket.request_id ||
        slot.ticket.source_id != source.index || !same_binding(slot.ticket.binding, source.binding)) {
      m.fail(SEEON_MEDIA_ERROR_RECORD_IDENTITY); return;
    }
    const auto phase = slot.phase.load();
    if (phase != RecordPhase::Starting && phase != RecordPhase::Active &&
        phase != RecordPhase::Ready && phase != RecordPhase::Consumed) {
      m.fail(SEEON_MEDIA_ERROR_RECORD_IDENTITY); return;
    }
    bool unclaimed = false;
    if (!slot.callback_claimed.compare_exchange_strong(unclaimed, true)) {
      m.late_record_callbacks.fetch_add(1); return;
    }
    if (phase == RecordPhase::Ready || phase == RecordPhase::Consumed)
      m.late_record_callbacks.fetch_add(1);
    auto &result = slot.callback_result;
    result = {}; result.ticket = slot.ticket;
    auto *info = static_cast<const NvDsSRRecordingInfo *>(info_pointer);
    result.result = SEEON_MEDIA_FATAL; result.error = SEEON_MEDIA_ERROR_RECORD_INFO;
    if (info) {
      result.ticket.session_id = info->sessionId; result.ticket.session_valid = 1;
      result.duration_ms = info->duration; result.width = info->width; result.height = info->height;
      result.contains_video = info->containsVideo != FALSE;
      result.contains_audio = info->containsAudio != FALSE;
      if (info->containerType == NVDSSR_CONTAINER_MP4 && info->containsVideo && info->width &&
          info->height && info->duration && copy_text(result.directory, info->dirpath) &&
          copy_text(result.filename, info->filename)) {
        result.result = SEEON_MEDIA_OK; result.error = SEEON_MEDIA_ERROR_NONE;
      }
    }
    // start-sr may complete synchronously. Session validation is deferred until
    // the action returns; a late writer never mutates the delivered result.
    slot.callback_completed = completed;
    slot.callback_done.store(true, std::memory_order_release);
  } catch (...) { m.fail(SEEON_MEDIA_ERROR_EXCEPTION); }
}
void recording_tick(SeeonMedia &m) {
  for (uint32_t index = 0; index < m.config.record_capacity; ++index) {
    std::unique_lock<std::mutex> lock(m.records_mutex);
    auto &slot = m.records[index];
    if (slot.phase.load() == RecordPhase::Unused) continue;
    complete_callback(m, slot);
    expire(m, slot);
    if (slot.phase.load() == RecordPhase::Queued) {
      slot.sdk_pending = true; slot.action_running = true;
      slot.phase.store(RecordPhase::Starting);
      NvDsSRSessionId session = UINT32_MAX;
      // Token publication precedes emission, and no registry/control lock is
      // held across a vendor call (including synchronous completion).
      lock.unlock();
      g_signal_emit_by_name(slot.source->element, "start-sr", &session,
                            guint(slot.lookback), guint(slot.forward), reinterpret_cast<gpointer>(slot.token));
      lock.lock();
      slot.action_running = false;
      if (session == UINT32_MAX) {
        m.warn(SEEON_MEDIA_ERROR_RECORD_START);
        deliver(slot, SEEON_MEDIA_FATAL, SEEON_MEDIA_ERROR_RECORD_START);
        slot.stop_requested.store(true);
        // No returned session is not proof of no SDK work. Quarantine until
        // proven quiescence; never rotate the whole owner to hide the slot.
      } else {
        slot.session.store(session);
        if (slot.phase.load() == RecordPhase::Starting) slot.phase.store(RecordPhase::Active);
      }
      complete_callback(m, slot);
      expire(m, slot);
    }
    if (slot.sdk_pending && slot.stop_requested.load() && !slot.stop_sent && slot.session.load() != kNoSession) {
      slot.stop_sent = true; slot.action_running = true;
      lock.unlock();
      g_signal_emit_by_name(slot.source->element, "stop-sr", guint(slot.session.load()));
      lock.lock();
      slot.action_running = false;
      complete_callback(m, slot);
      // An accepted stop is not retirement: preserve source + reservation.
    }
    reap(m, index);
  }
}
void recording_cancel(SeeonMedia &m) {
  // Only after NULL, all producer retirement acknowledgements, and drained
  // callback leases. Those facts retire even an unresolved/failed start action.
  std::lock_guard<std::mutex> lock(m.records_mutex);
  for (uint32_t index = 0; index < m.config.record_capacity; ++index) {
    auto &slot = m.records[index];
    if (slot.phase.load() == RecordPhase::Unused) continue;
    complete_callback(m, slot);
    deliver(slot, SEEON_MEDIA_STALE, SEEON_MEDIA_ERROR_CANCELLED);
    slot.sdk_pending = false; slot.action_running = false;
    reap(m, index);
  }
}
} // namespace seeon_media

extern "C" SeeonMediaResult seeon_media_record_start(SeeonMedia *m, uint32_t source,
    const SeeonMediaBinding *binding, uint64_t request, uint32_t lookback,
    uint32_t forward, SeeonMediaRecordTicket *out, size_t bytes) {
  return api(m, [&]() -> SeeonMediaResult {
    if (!out || bytes < sizeof(*out)) return SEEON_MEDIA_TOO_SMALL;
    auto status = source_check(*m, source, binding);
    if (status != SEEON_MEDIA_OK) return status;
    auto &s = m->sources[source];
    if (!s.rtsp) return SEEON_MEDIA_UNSUPPORTED;
    status = admission(*m);
    if (status != SEEON_MEDIA_OK) return status;
    if (!request || request <= m->last_record_request || !forward ||
        lookback >= m->config.record_cache_seconds) return SEEON_MEDIA_STALE;
    if (!s.linked.load() || !s.frames.load()) return SEEON_MEDIA_BUSY;
    std::unique_lock<std::mutex> lock(m->records_mutex, std::try_to_lock);
    if (!lock.owns_lock()) return SEEON_MEDIA_BUSY;
    for (uint32_t index = 0; index < m->config.record_capacity; ++index) {
      complete_callback(*m, m->records[index]);
      expire(*m, m->records[index]);
      reap(*m, index);
    }
    status = admission(*m);
    if (status != SEEON_MEDIA_OK) return status;
    const int active = s.active_record.load();
    if (active >= 0) {
      auto &slot = m->records[active];
      expire(*m, slot);
      const auto phase = slot.phase.load();
      if ((phase != RecordPhase::Queued && phase != RecordPhase::Starting && phase != RecordPhase::Active) ||
          slot.stop_requested.load()) return m->capacity();
      // Preserve one healthy in-flight recording, not a failed/quarantined one.
      *out = ticket(slot, true);
      m->last_record_request = request;
      return SEEON_MEDIA_OK;
    }
    uint32_t index = 0;
    while (index < m->config.record_capacity && m->records[index].phase.load() != RecordPhase::Unused) ++index;
    if (index == m->config.record_capacity) return m->capacity();
    const auto token = allocate_token();
    if (!token) { m->fail(SEEON_MEDIA_ERROR_CAPACITY); return SEEON_MEDIA_FATAL; }
    auto &slot = m->records[index];
    auto &registration = m->lifetime->records[index];
    slot.owner = m; slot.source = &s; slot.registration = &registration; slot.token = token;
    slot.ticket = {*binding, request, source, 0, 0, 0};
    slot.lookback = lookback; slot.forward = forward;
    slot.deadline = Clock::now() + std::chrono::seconds(uint64_t(forward) + 30);
    slot.session.store(kNoSession);
    slot.stop_requested.store(false); slot.callback_claimed.store(false); slot.callback_done.store(false);
    slot.sdk_pending = false; slot.action_running = false; slot.stop_sent = false; slot.callback_processed = false;
    slot.callback_result = {}; slot.callback_completed = {}; slot.result = {};
    registration.owner = m; registration.source = &s; registration.slot = &slot; registration.request = request;
    registration.token.store(token);
    registration.entries.state.store(0);
    m->last_record_request = request;
    m->records_used.fetch_add(1);
    s.active_record.store(int(index));
    *out = slot.ticket;
    slot.phase.store(RecordPhase::Queued);
    return SEEON_MEDIA_OK;
  });
}
extern "C" SeeonMediaResult seeon_media_record_stop(SeeonMedia *m, uint32_t source,
    const SeeonMediaBinding *binding, uint64_t request, uint32_t session) {
  return api(m, [&]() -> SeeonMediaResult {
    const auto status = source_check(*m, source, binding);
    if (status != SEEON_MEDIA_OK) return status;
    auto &s = m->sources[source];
    if (!s.rtsp) return SEEON_MEDIA_UNSUPPORTED;
    if (m->stop_requested.load()) return SEEON_MEDIA_STALE;
    std::unique_lock<std::mutex> lock(m->records_mutex, std::try_to_lock);
    if (!lock.owns_lock()) return SEEON_MEDIA_BUSY;
    const int active = s.active_record.load();
    if (active < 0) return SEEON_MEDIA_STALE;
    auto &slot = m->records[active];
    if (request != slot.ticket.request_id) return SEEON_MEDIA_STALE;
    if (slot.session.load() == kNoSession) return SEEON_MEDIA_BUSY;
    if (slot.session.load() != session || !slot.sdk_pending) return SEEON_MEDIA_STALE;
    slot.stop_requested.store(true);
    return SEEON_MEDIA_OK;
  });
}
extern "C" SeeonMediaResult seeon_media_try_read_record(SeeonMedia *m, SeeonMediaRecord *out, size_t bytes) {
  return api(m, [&]() -> SeeonMediaResult {
    if (!out || bytes < sizeof(*out)) return SEEON_MEDIA_TOO_SMALL;
    std::unique_lock<std::mutex> lock(m->records_mutex, std::try_to_lock);
    if (!lock.owns_lock()) return SEEON_MEDIA_BUSY;
    for (uint32_t index = 0; index < m->config.record_capacity; ++index) {
      auto &slot = m->records[index];
      // The caller can deliver a deadline while control is blocked in start-sr;
      // the immutable ticket and pending SDK operation still retain the slot.
      complete_callback(*m, slot);
      expire(*m, slot);
      reap(*m, index);
      if (slot.phase.load() != RecordPhase::Ready) continue;
      *out = slot.result;
      slot.phase.store(RecordPhase::Consumed);
      reap(*m, index);
      return SEEON_MEDIA_OK;
    }
    return m->fatal.load() ? SEEON_MEDIA_FATAL : SEEON_MEDIA_EMPTY;
  });
}
