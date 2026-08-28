#include "au_transport.hpp"

#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <utility>

namespace seeon {
namespace {
constexpr std::array<char, 4> kMagic{'S', 'A', 'U', '1'};
constexpr std::uint8_t kAccessUnit = 1;
constexpr std::uint8_t kGap = 2;

std::size_t envelope_bytes(const AuEnvelope& envelope) {
  return envelope.camera.size() + envelope.unit.parser_caps.size() +
         envelope.unit.payload.size() + envelope.unit.codec_data.size();
}
#pragma pack(push, 1)
struct Header {
  std::array<char, 4> magic;
  std::uint32_t body_size;
  std::uint8_t kind;
  std::uint8_t codec;
  std::uint8_t framing;
  std::uint8_t keyframe;
  std::uint32_t generation;
  std::uint64_t epoch;
  std::uint64_t sequence;
  std::int64_t pts;
  std::int64_t dts;
  std::int64_t duration;
  std::int32_t time_base_num;
  std::int32_t time_base_den;
  std::uint32_t width;
  std::uint32_t height;
  std::uint16_t camera_size;
  std::uint16_t caps_size;
  std::uint32_t codec_data_size;
  std::uint32_t payload_size;
};
#pragma pack(pop)
static_assert(sizeof(Header) == 84);

std::vector<std::uint8_t> encode(const AuEnvelope& envelope, bool gap) {
  const auto& unit = envelope.unit;
  Header header{kMagic,
                0,
                gap ? kGap : kAccessUnit,
                static_cast<std::uint8_t>(unit.codec),
                static_cast<std::uint8_t>(unit.framing),
                static_cast<std::uint8_t>(unit.keyframe),
                envelope.generation,
                envelope.epoch,
                envelope.sequence,
                unit.pts,
                unit.dts,
                unit.duration,
                unit.time_base_num,
                unit.time_base_den,
                unit.width,
                unit.height,
                static_cast<std::uint16_t>(envelope.camera.size()),
                static_cast<std::uint16_t>(unit.parser_caps.size()),
                gap ? 0U : static_cast<std::uint32_t>(unit.codec_data.size()),
                gap ? 0U : static_cast<std::uint32_t>(unit.payload.size())};
  header.body_size = header.camera_size + header.caps_size + header.codec_data_size +
                     header.payload_size;
  std::vector<std::uint8_t> bytes(sizeof(header) + header.body_size);
  std::memcpy(bytes.data(), &header, sizeof(header));
  auto* output = bytes.data() + sizeof(header);
  if (header.camera_size > 0) {
    std::memcpy(output, envelope.camera.data(), header.camera_size);
    output += header.camera_size;
  }
  if (header.caps_size > 0) {
    std::memcpy(output, unit.parser_caps.data(), header.caps_size);
    output += header.caps_size;
  }
  if (!gap && header.codec_data_size > 0) {
    std::memcpy(output, unit.codec_data.data(), header.codec_data_size);
    output += header.codec_data_size;
  }
  if (!gap && header.payload_size > 0) {
    std::memcpy(output, unit.payload.data(), header.payload_size);
  }
  return bytes;
}

bool send_all(int descriptor, const std::vector<std::uint8_t>& bytes) {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const auto count = send(descriptor, bytes.data() + offset, bytes.size() - offset, MSG_NOSIGNAL);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return false;
    offset += static_cast<std::size_t>(count);
  }
  return true;
}
}  // namespace

PipelineBinding::PipelineBinding(std::uint32_t generation, std::uint64_t epoch)
    : generation_(generation), epoch_(epoch) {}
void PipelineBinding::invalidate() { std::lock_guard lock{mutex_}; live_ = false; }
bool PipelineBinding::dispatch_au(
    const std::function<void(std::uint32_t, std::uint64_t, std::uint64_t)>& action) {
  std::lock_guard lock{mutex_};
  if (!live_) return false;
  action(generation_, epoch_, ++sequence_);
  return true;
}
bool PipelineBinding::dispatch_frame(
    const std::function<void(std::uint32_t, std::uint64_t)>& action) {
  std::lock_guard lock{mutex_};
  if (!live_) return false;
  action(generation_, epoch_);
  return true;
}

AuSender::AuSender(int descriptor, std::size_t max_items, std::size_t max_bytes)
    : descriptor_(descriptor), max_items_(max_items), max_bytes_(max_bytes),
      thread_(&AuSender::run, this) {}
AuSender::~AuSender() { stop(); }

bool AuSender::enqueue(AuEnvelope envelope) {
  const auto size = envelope_bytes(envelope);
  std::lock_guard lock{mutex_};
  if (stopped_) return false;
  // Validity limits are absolute: an envelope that cannot legally be framed is
  // refused outright and is never eligible for reservation, because a reserved
  // unit is later emitted as an ordinary access unit and would otherwise
  // bypass the very limit that rejected it.
  // Identity fields that overflow the wire's uint16 lengths cannot be encoded
  // even as a gap marker, so such an envelope is dropped outright below.
  const bool identity_too_long = envelope.camera.size() > UINT16_MAX ||
                                 envelope.unit.parser_caps.size() > UINT16_MAX;
  const bool invalid = size > kMaxAuFrameBytes || size > max_bytes_ || identity_too_long;
  const auto existing = transitions_.find(envelope.camera);
  // No ordering barrier is needed for a camera with a pending reservation:
  // run() sends the reservation before anything in queue_, re-checking on
  // every wake, so a unit admitted now always follows it on the wire.
  const bool congested =
      gap_.has_value() || queue_.size() >= max_items_ || bytes_ + size > max_bytes_;
  if (invalid || congested) {
    ++dropped_;
    // A unit that opens a stream epoch is reserved rather than destroyed, but
    // only when it is legally framable and was refused for congestion. A
    // reservation is later emitted as an ORDINARY access unit, so admitting an
    // invalid one here would bypass the limit that rejected it.
    //
    // Held per camera and keyed by (generation, epoch) so a higher generation
    // wins even though its epoch restarts at 1 - this sender is shared by every
    // camera, so a scalar epoch comparison could not express that. Delivering
    // it as an ordinary unit is what makes it effective: the receiver adopts an
    // epoch from a normal sequence-1 unit and discards gap markers before
    // adoption.
    if (!invalid && envelope.sequence == 1) {
      const bool newer =
          existing == transitions_.end() ||
          std::make_pair(envelope.generation, envelope.epoch) >
              std::make_pair(existing->second.generation, existing->second.epoch);
      // Budget the projected total AFTER replacement. Charging the newcomer
      // before crediting the reservation it replaces would reject a newer
      // transition that plainly fits, and it would then fall through to the
      // gap slot - which the receiver discards before adoption, reintroducing
      // exactly the stranding this reservation exists to prevent.
      std::size_t replaced = 0;
      if (existing != transitions_.end()) replaced = envelope_bytes(existing->second);
      // Reservations need their own allowance, not a share of the queue's.
      // A transition is refused precisely BECAUSE the queue is full or its
      // bytes are exhausted, so budgeting it against the same total would make
      // it unreservable in exactly the situation the reservation exists for.
      //
      // The allowance is deliberately small - an eighth of the aggregate - so
      // queued bytes plus reserved bytes are capped at max_bytes_ * 9/8 rather
      // than the 2x a second full budget would permit. That is a bound on
      // those two pools only: gap_, the envelope the sender has popped and is
      // still writing, and its encoded copy are all uncharged, so it is not a
      // whole-sender ceiling. And fitting within the 1/8 makes a transition
      // ELIGIBLE, not guaranteed: the allowance is aggregate across cameras
      // and may already be consumed, and reservation still depends on the
      // refusal and newness conditions above.
      const std::size_t transition_allowance = max_bytes_ / 8;
      const std::size_t projected = transition_bytes_ - replaced + size;
      if (newer && projected <= transition_allowance) {
        transition_bytes_ = projected;
        transitions_[envelope.camera] = std::move(envelope);
        ready_.notify_one();
        return false;
      }
      if (!newer) {
        ready_.notify_one();
        return false;
      }
    }
    if (!gap_.has_value() && !identity_too_long) gap_ = std::move(envelope);
    ready_.notify_one();
    return false;
  }
  bytes_ += size;
  queue_.push_back(std::move(envelope));
  ready_.notify_one();
  return true;
}

void AuSender::run() {
  while (true) {
    AuEnvelope envelope{};
    bool gap = false;
    {
      std::unique_lock lock{mutex_};
      ready_.wait(lock, [this] {
        return stopped_ || gap_.has_value() || !queue_.empty() || !transitions_.empty();
      });
      if (stopped_ && !gap_.has_value() && queue_.empty() && transitions_.empty()) return;
      if (!transitions_.empty()) {
        // First, ahead of the queue and the gap, and as an ordinary unit: the
        // receiver adopts an epoch from a normal sequence-1 access unit, never
        // from a gap marker. A reservation exists because the sender is
        // congested, so draining it only once the queue empties starves that
        // camera for as long as the other twelve keep the queue non-empty.
        // Units of the superseded epoch still queued behind it are dropped by
        // the receiver as debris, which is what they are.
        const auto first = transitions_.begin();
        envelope = std::move(first->second);
        transition_bytes_ -= envelope_bytes(envelope);
        transitions_.erase(first);
      } else if (!queue_.empty()) {
        envelope = std::move(queue_.front());
        bytes_ -= envelope_bytes(envelope);
        queue_.pop_front();
      } else {
        envelope = std::move(*gap_);
        gap_.reset();
        gap = true;
      }
    }
    if (!send_all(descriptor_, encode(envelope, gap))) {
      std::lock_guard lock{mutex_};
      stopped_ = true;
      queue_.clear();
      gap_.reset();
      transitions_.clear();
      transition_bytes_ = 0;
      return;
    }
  }
}

void AuSender::stop() {
  bool shutdown_required = false;
  {
    std::lock_guard lock{mutex_};
    shutdown_required = !stopped_;
    stopped_ = true;
  }
  if (shutdown_required) shutdown(descriptor_, SHUT_WR);
  ready_.notify_all();
  if (thread_.joinable()) thread_.join();
}
std::uint64_t AuSender::dropped() const { std::lock_guard lock{mutex_}; return dropped_; }
}  // namespace seeon
