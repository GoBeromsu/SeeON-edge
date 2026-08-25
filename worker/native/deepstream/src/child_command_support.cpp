#include "child_command_support.hpp"

#include "perception_wire.hpp"

#include <sys/socket.h>

#include <cstring>
#include <utility>

namespace seeon::command_support {
namespace {
#pragma pack(push, 1)
struct StatusPayload {
  std::uint64_t metadata_published;
  std::uint64_t metadata_overwritten;
  std::uint64_t wake_dropped;
  std::uint64_t source_failures;
  std::uint64_t malformed_frames;
  std::uint32_t source_count;
  std::uint8_t custom_transform_available;
};
#pragma pack(pop)

void wake_metadata(ServerState& state, const std::string& camera) {
  const auto sent = send(state.options.wake_fd, camera.data(), camera.size(), MSG_DONTWAIT);
  if (sent < 0) {
    ++state.wake_dropped;
  }
}
}  // namespace

std::vector<std::uint8_t> text_payload(const std::string& value) {
  return {value.begin(), value.end()};
}

ipc::Message error_reply(const ipc::Message& request, const std::string& detail) {
  return ipc::reply(request, ipc::Kind::kError, text_payload(detail));
}

bool identity_matches(const ServerState& state, const ipc::Message& request) {
  return request.header.worker_boot_id == state.options.worker_boot_id &&
         request.header.child_instance_id == state.options.child_instance_id &&
         !request.camera.empty() && request.camera.size() <= 128 &&
         request.transform == "seeon-perception-v1";
}

void publish_metadata(ServerState& state, const ipc::Message& request, SourceSlot& source) {
  ++source.source_sequence;
  ++state.publish_sequence;
  ipc::Message metadata = request;
  metadata.header.kind = static_cast<std::uint8_t>(ipc::Kind::kMetadata);
  metadata.header.stream_epoch = source.epoch;
  metadata.header.source_pts = request.header.source_pts == 0
                                   ? source.source_sequence * 1000
                                   : request.header.source_pts;
  metadata.header.source_sequence = source.source_sequence;
  metadata.header.native_publish_sequence = state.publish_sequence;
  metadata.header.request_id = 0;
  metadata.payload = encode_empty_perception(metadata);
  const bool wake_required = !source.latest.has_value();
  if (!wake_required) {
    ++state.overwritten;
  }
  source.latest = std::move(metadata);
  {
    std::lock_guard published_lock{state.published_mutex};
    ++state.published;
  }
  if (wake_required) {
    wake_metadata(state, request.camera);
  }
  state.published_condition.notify_all();
}

ipc::Message status_reply(const ServerState& state, const ipc::Message& request) {
  const StatusPayload status{
      state.published.load(),
      state.overwritten,
      state.wake_dropped,
      state.source_failures,
      state.malformed,
      static_cast<std::uint32_t>(state.runtime.count()),
      static_cast<std::uint8_t>(state.runtime.custom_transform_available()),
  };
  std::vector<std::uint8_t> payload(sizeof(status));
  std::memcpy(payload.data(), &status, sizeof(status));
  return ipc::reply(request, ipc::Kind::kStatusReply, std::move(payload));
}
}  // namespace seeon::command_support

namespace seeon {
void ServerState::on_access_unit(const std::string& camera,
                                 const PipelineBindingPtr& binding,
                                 ParsedAccessUnit unit) {
  static_cast<void>(binding->dispatch_au(
      [this, &camera, &unit](std::uint32_t generation, std::uint64_t epoch,
                             std::uint64_t sequence) {
        static_cast<void>(au_sender.enqueue(
            AuEnvelope{camera, generation, epoch, sequence, std::move(unit)}));
      }));
}

void ServerState::on_frame(const std::string& camera, const PipelineBindingPtr& binding,
                           std::uint64_t pts) {
  static_cast<void>(binding->dispatch_frame(
      [this, &camera, pts, &binding](std::uint32_t generation, std::uint64_t epoch) {
        std::lock_guard lock{slot_mutex};
        const auto found = sources.find(camera);
        if (found == sources.end() || found->second.binding != binding) return;
        ipc::Message message{};
        message.header.worker_boot_id = options.worker_boot_id;
        message.header.child_instance_id = options.child_instance_id;
        message.header.source_generation = generation;
        message.header.stream_epoch = epoch;
        message.header.source_pts = pts;
        message.camera = camera;
        message.transform = "seeon-perception-v1";
        command_support::publish_metadata(*this, message, found->second);
      }));
}
}  // namespace seeon
