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
  std::memcpy(output, envelope.camera.data(), header.camera_size);
  output += header.camera_size;
  std::memcpy(output, unit.parser_caps.data(), header.caps_size);
  output += header.caps_size;
  if (!gap) {
    std::memcpy(output, unit.codec_data.data(), header.codec_data_size);
    output += header.codec_data_size;
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
  const auto size = envelope.camera.size() + envelope.unit.parser_caps.size() +
                    envelope.unit.payload.size() + envelope.unit.codec_data.size();
  std::lock_guard lock{mutex_};
  if (stopped_) return false;
  if (gap_.has_value() || size > kMaxAuFrameBytes || size > max_bytes_ ||
      envelope.camera.size() > UINT16_MAX || envelope.unit.parser_caps.size() > UINT16_MAX ||
      queue_.size() >= max_items_ || bytes_ + size > max_bytes_) {
    ++dropped_;
    if (!gap_.has_value()) gap_ = std::move(envelope);
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
      ready_.wait(lock, [this] { return stopped_ || gap_.has_value() || !queue_.empty(); });
      if (stopped_ && !gap_.has_value() && queue_.empty()) return;
      if (!queue_.empty()) {
        envelope = std::move(queue_.front());
        bytes_ -= envelope.camera.size() + envelope.unit.parser_caps.size() +
                  envelope.unit.payload.size() + envelope.unit.codec_data.size();
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
