#include "media_internal.h"

#ifdef SEEON_PRIVATE_N2_WITHHELD
#include <cerrno>
#include <cstdio>
#include <fcntl.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>
#endif

using namespace seeon_media;
namespace {
#ifdef SEEON_PRIVATE_N2_WITHHELD
// Private publish-once protocol in the fixture's fresh, owned record directory.
// No SDK pointers, native state writes, locks, or release controls belong here.
struct PrivateN2Fd {
  int value;
  explicit PrivateN2Fd(int fd) noexcept : value(fd) {}
  ~PrivateN2Fd() { if (value >= 0) ::close(value); }
  bool close() noexcept {
    const int fd = value;
    value = -1;
    return ::close(fd) == 0;
  }
  PrivateN2Fd(const PrivateN2Fd &) = delete;
  PrivateN2Fd &operator=(const PrivateN2Fd &) = delete;
};
enum class PrivateN2Read { Absent, Invalid, Valid };
template<size_t Fields>
PrivateN2Read private_n2_read(int directory, const char *name, const char (&magic)[9],
                            std::array<uint64_t, Fields> &fields) noexcept {
  static_assert(Fields == 7 || Fields == 4, "Only bounded arm/query inputs");
  constexpr size_t size = 8 + 8 * Fields;
  PrivateN2Fd file(::openat(directory, name, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
  if (file.value < 0) return errno == ENOENT ? PrivateN2Read::Absent : PrivateN2Read::Invalid;
  struct stat status{};
  if (::fstat(file.value, &status) != 0 || !S_ISREG(status.st_mode) ||
      status.st_uid != ::geteuid() || (status.st_mode & (S_IWGRP | S_IWOTH)) ||
      status.st_size != static_cast<off_t>(size)) return PrivateN2Read::Invalid;
  std::array<unsigned char, size + 1> bytes{};
  size_t used = 0;
  while (used < bytes.size()) {
    const auto count = ::read(file.value, bytes.data() + used, bytes.size() - used);
    if (count < 0) return PrivateN2Read::Invalid;
    if (!count) break;
    used += static_cast<size_t>(count);
  }
  if (used != size || std::memcmp(bytes.data(), magic, 8) != 0 || !file.close())
    return PrivateN2Read::Invalid;
  for (size_t index = 0; index < Fields; ++index) {
    uint64_t value = 0;
    for (size_t byte = 0; byte < 8; ++byte)
      value |= uint64_t(bytes[8 + 8 * index + byte]) << (8 * byte);
    fields[index] = value;
  }
  return PrivateN2Read::Valid;
}
template<size_t Fields>
bool private_n2_publish(int directory, const char *temporary, const char *name,
                        const char (&magic)[9], const std::array<uint64_t, Fields> &fields) noexcept {
  static_assert(Fields == 13 || Fields == 20, "Only bounded entry/witness outputs");
  std::array<unsigned char, 8 + 8 * Fields> bytes{};
  std::memcpy(bytes.data(), magic, 8);
  for (size_t index = 0; index < Fields; ++index) {
    for (size_t byte = 0; byte < 8; ++byte)
      bytes[8 + 8 * index + byte] = static_cast<unsigned char>(fields[index] >> (8 * byte));
  }
  PrivateN2Fd file(::openat(directory, temporary,
      O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK, 0600));
  if (file.value < 0) return false;
  struct stat status{};
  if (::fstat(file.value, &status) != 0 || !S_ISREG(status.st_mode) ||
      status.st_uid != ::geteuid() || status.st_size != 0) return false;
  size_t written = 0;
  while (written < bytes.size()) {
    const auto count = ::write(file.value, bytes.data() + written, bytes.size() - written);
    if (count <= 0) return false;
    written += static_cast<size_t>(count);
  }
  if (::fstat(file.value, &status) != 0 || status.st_size != static_cast<off_t>(bytes.size()) ||
      ::fsync(file.value) != 0 || !file.close()) return false;
  // Publication is the final fallible operation: never overwrite or truncate.
  return ::renameat2(directory, temporary, directory, name, RENAME_NOREPLACE) == 0;
}
void private_n2_pause() noexcept {
  const struct timespec pacing{0, 10000000};
  (void)::nanosleep(&pacing, nullptr);
}
[[noreturn]] void private_n2_park() noexcept {
  for (;;) private_n2_pause();
}
[[noreturn]] void private_n2_hold(int directory, const std::array<uint64_t, 13> &entry,
                                const SeeonMedia &m, const RecordSlot &slot,
                                const CallbackGuard &guard, const RecordLease &lease) noexcept {
  // Entry is irrevocable even if I/O fails. The caller's real stack guards and
  // retained directory FD stay alive; missing evidence makes the fixture fail.
  try {
    if (!private_n2_publish(directory, ".n2-entry.tmp", ".n2-entry.bin", "N2ENT001", entry))
      private_n2_park();
    std::array<uint64_t, 4> query{};
    for (;;) {
      const auto read = private_n2_read(directory, ".n2-query.bin", "N2QRY001", query);
      if (read == PrivateN2Read::Absent) { private_n2_pause(); continue; }
      if (read != PrivateN2Read::Valid || query[0] != entry[0] || query[1] != entry[1] ||
          !(query[2] || query[3]) || (query[2] == entry[0] && query[3] == entry[1]))
        private_n2_park();
      break;
    }
    std::array<uint64_t, 20> witness{};
    for (size_t index = 0; index < entry.size(); ++index) witness[index] = entry[index];
    witness[13] = query[2];
    witness[14] = query[3];
    // Bounded atomic observations, not a simultaneous transactional snapshot.
    witness[15] = m.records_used.load(std::memory_order_acquire);
    witness[16] = slot.callback_done.load(std::memory_order_acquire);
    witness[17] = guard.registration->entries.state.load(std::memory_order_acquire) & ~LeaseGate::retired;
    witness[18] = lease.registration->entries.state.load(std::memory_order_acquire) & ~LeaseGate::retired;
    witness[19] = guard.registration->lifetime->entries.state.load(std::memory_order_acquire) & ~LeaseGate::retired;
    (void)private_n2_publish(directory, ".n2-witness.tmp", ".n2-witness.bin", "N2WIT001", witness);
  } catch (...) {
    // No exception may reach record_done's ordinary diagnostic/release path.
  }
  private_n2_park();
}
void private_n2_withhold(const SeeonMedia &m, const RecordSlot &slot,
                         const CallbackGuard &guard, const RecordLease &lease) noexcept {
  const auto &result = slot.callback_result; // Already copied genuine SDK data only.
  const auto session = slot.session.load(std::memory_order_acquire);
  if (result.result != SEEON_MEDIA_OK || result.error != SEEON_MEDIA_ERROR_NONE ||
      result.ticket.session_valid != 1 || session == kNoSession ||
      session != result.ticket.session_id || !result.duration_ms || !result.width || !result.height ||
      result.contains_video != 1 || result.contains_audio > 1) return;
  PrivateN2Fd directory(::open(m.record_directory.c_str(),
      O_RDONLY | O_DIRECTORY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK));
  if (directory.value < 0) return;
  struct stat status{};
  if (::fstat(directory.value, &status) != 0 || !S_ISDIR(status.st_mode) ||
      status.st_uid != ::geteuid() || (status.st_mode & (S_IWGRP | S_IWOTH))) return;
  std::array<uint64_t, 7> arm{};
  if (private_n2_read(directory.value, ".n2-arm.bin", "N2ARM001", arm) != PrivateN2Read::Valid ||
      !(arm[0] || arm[1]) || arm[2] != result.ticket.source_id ||
      arm[3] != result.ticket.binding.token || arm[4] != result.ticket.binding.generation ||
      arm[5] != result.ticket.binding.epoch || arm[6] != result.ticket.request_id) return;
  // Freshness is the fixture's exclusively created directory plus absence of
  // all later fixed protocol names. No scans or nonce history are retained.
  constexpr const char *later_names[] = {".n2-entry.tmp", ".n2-entry.bin", ".n2-query.tmp",
      ".n2-query.bin", ".n2-witness.tmp", ".n2-witness.bin"};
  for (const auto *name : later_names) {
    if (::fstatat(directory.value, name, &status, AT_SYMLINK_NOFOLLOW) == 0 || errno != ENOENT) return;
  }
  const std::array<uint64_t, 13> entry{arm[0], arm[1], result.ticket.source_id,
      result.ticket.binding.token, result.ticket.binding.generation, result.ticket.binding.epoch,
      result.ticket.request_id, session, result.duration_ms, result.width, result.height,
      result.contains_video, result.contains_audio};
  // Never wait for an arm or session. Past this boundary there is no return.
  private_n2_hold(directory.value, entry, m, slot, guard, lease);
}
#endif

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
// A started session is not finished until sr-done or proven NULL retirement.
// Publishing Stale here would win first-writer-wins and discard that receipt.
bool recording_started(const RecordSlot &slot) noexcept {
  return slot.sdk_pending && slot.session.load() != kNoSession;
}
void expire(SeeonMedia &m, RecordSlot &slot) noexcept {
  const auto phase = slot.phase.load();
  if (phase == RecordPhase::Unused || phase == RecordPhase::Ready || phase == RecordPhase::Consumed) return;
  if (m.fatal.load()) {
    deliver(slot, SEEON_MEDIA_FATAL, SeeonMediaError(unpack(m.fatal.load()).code));
  } else if (recording_started(slot) && Clock::now() < slot.deadline) {
    if (m.stop_requested.load()) slot.stop_requested.store(true);
    return; // Leave the vendor callback as the only genuine receipt writer.
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
#ifdef SEEON_PRIVATE_N2_WITHHELD
    private_n2_withhold(m, slot, guard, lease);
#endif
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
    // stop-sr was accepted and sr-done never arrived before NULL. The Stale
    // receipt stays empty; this warning is the only evidence of that miss.
    if (slot.stop_sent && slot.sdk_pending) m.warn(SEEON_MEDIA_ERROR_RECORD_TIMEOUT);
    deliver(slot, SEEON_MEDIA_STALE, SEEON_MEDIA_ERROR_CANCELLED);
    slot.sdk_pending = false; slot.action_running = false;
    reap(m, index);
  }
}
bool recording_pending(SeeonMedia &m) {
  std::lock_guard<std::mutex> lock(m.records_mutex);
  for (uint32_t index = 0; index < m.config.record_capacity; ++index) {
    const auto &slot = m.records[index];
    const auto phase = slot.phase.load();
    if (recording_started(slot) &&
        (phase == RecordPhase::Starting || phase == RecordPhase::Active)) return true;
  }
  return false;
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
