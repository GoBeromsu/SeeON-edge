#pragma once

#include <array>
#include <cstdint>
#include <cstring>
#include <optional>
#include <string>
#include <vector>

namespace seeon::ipc {
constexpr std::array<char, 4> kMagic{'S', 'D', 'S', '1'};
constexpr std::uint8_t kVersion = 1;

enum class Kind : std::uint8_t {
  kAddSource = 1,
  kRemoveSource = 2,
  kRecord = 3,
  kSnapshot = 4,
  kStatus = 5,
  kSourceFailure = 6,
  kFatal = 7,
  kEmitMetadata = 8,
  kGetLatest = 9,
  kShutdown = 10,
  kWaitPublish = 11,
  kGetSourceState = 12,
  kInjectSourceEos = 13,
  kSetPreviewDemand = 14,
  kGetPreviewStatus = 15,
  kWaitPreview = 16,
  kAck = 64,
  kStatusReply = 65,
  kError = 66,
  kEpochStarted = 67,
  kCapabilityInactive = 68,
  kMetadata = 128,
};

#pragma pack(push, 1)
struct Header {
  std::array<char, 4> magic;
  std::uint8_t version;
  std::uint8_t kind;
  std::uint16_t flags;
  std::uint32_t body_size;
  std::uint32_t source_generation;
  std::uint64_t stream_epoch;
  std::uint64_t source_pts;
  std::uint64_t source_sequence;
  std::uint64_t native_publish_sequence;
  std::uint64_t request_id;
  std::array<std::uint8_t, 16> worker_boot_id;
  std::array<std::uint8_t, 16> child_instance_id;
  std::uint16_t camera_size;
  std::uint16_t transform_size;
};
#pragma pack(pop)
static_assert(sizeof(Header) == 92);

struct Message {
  Header header;
  std::string camera;
  std::string transform;
  std::vector<std::uint8_t> payload;
};

inline std::optional<Message> decode(const std::vector<std::uint8_t>& bytes) {
  if (bytes.size() < sizeof(Header)) {
    return std::nullopt;
  }
  Header header{};
  std::memcpy(&header, bytes.data(), sizeof(header));
  if (header.magic != kMagic || header.version != kVersion ||
      bytes.size() != sizeof(Header) + header.body_size ||
      static_cast<std::uint32_t>(header.camera_size) + header.transform_size > header.body_size) {
    return std::nullopt;
  }
  const auto* body = bytes.data() + sizeof(Header);
  Message message{header, {}, {}, {}};
  message.camera.assign(reinterpret_cast<const char*>(body), header.camera_size);
  message.transform.assign(reinterpret_cast<const char*>(body + header.camera_size),
                           header.transform_size);
  const auto payload_offset = static_cast<std::size_t>(header.camera_size) + header.transform_size;
  message.payload.assign(body + payload_offset, body + header.body_size);
  return message;
}

inline std::vector<std::uint8_t> encode(const Message& message) {
  Header header = message.header;
  header.magic = kMagic;
  header.version = kVersion;
  header.camera_size = static_cast<std::uint16_t>(message.camera.size());
  header.transform_size = static_cast<std::uint16_t>(message.transform.size());
  header.body_size = static_cast<std::uint32_t>(message.camera.size() + message.transform.size() +
                                                message.payload.size());
  std::vector<std::uint8_t> bytes(sizeof(Header) + header.body_size);
  std::memcpy(bytes.data(), &header, sizeof(header));
  auto* body = bytes.data() + sizeof(Header);
  std::memcpy(body, message.camera.data(), message.camera.size());
  std::memcpy(body + message.camera.size(), message.transform.data(), message.transform.size());
  std::memcpy(body + message.camera.size() + message.transform.size(), message.payload.data(),
              message.payload.size());
  return bytes;
}

inline Message reply(const Message& request, Kind kind, std::vector<std::uint8_t> payload = {}) {
  Message response = request;
  response.header.kind = static_cast<std::uint8_t>(kind);
  response.payload = std::move(payload);
  return response;
}
}  // namespace seeon::ipc
