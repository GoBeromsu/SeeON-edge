#include "child_command_support.hpp"

#include <cstdio>

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

namespace {

std::uint8_t channel_state(std::size_t count) { return count == 0 ? 2 : 1; }

PerceptionPayload perception_payload(const ServerState& state, SourceSlot& source,
                                     const trt::PerceptionResult& result,
                                     std::uint64_t source_time_ns) {
  PerceptionPayload payload;
  payload.source_width = static_cast<std::uint16_t>(result.source_width);
  payload.source_height = static_cast<std::uint16_t>(result.source_height);
  payload.source_time_ns = source_time_ns;
  const auto& cues = state.options.person_box_from_person_engine ? result.person
                                                                 : result.pose.boxes;
  payload.person_state = channel_state(cues.size());
  payload.boxes.reserve(cues.size());
  for (const auto& box : cues) {
    payload.boxes.push_back(WireBox{box.x1, box.y1, box.x2, box.y2, box.confidence});
  }
  payload.pose_state = channel_state(result.pose.poses.size());
  payload.poses.reserve(result.pose.poses.size());
  for (const auto& pose : result.pose.poses) {
    std::vector<WireKeypoint> points;
    points.reserve(pose.size());
    for (const auto& point : pose) {
      points.push_back(WireKeypoint{point.x, point.y, point.score});
    }
    payload.poses.push_back(std::move(points));
  }
  payload.bed_state = channel_state(result.bed.size());
  payload.bed_regions.reserve(result.bed.size());
  for (const auto& region : result.bed) {
    WireBedRegion wire{WireBox{region.bounds.x1, region.bounds.y1, region.bounds.x2,
                               region.bounds.y2, region.bounds.confidence},
                       {}};
    wire.polygon.reserve(region.polygon.size());
    for (const auto& [x, y] : region.polygon) wire.polygon.emplace_back(x, y);
    payload.bed_regions.push_back(std::move(wire));
  }
  if (source.association == nullptr) {
    source.association = std::make_shared<perception::LegacyGreedyBboxIou>();
  }
  std::vector<perception::ParsedBox> association_cues;
  association_cues.reserve(cues.size());
  for (const auto& box : cues) association_cues.push_back(box);
  const auto association = source.association->observe(association_cues);
  WireAssociation wire_association{"legacy-greedy-bbox-iou.v1", "person_box", {}, {}};
  wire_association.selections.reserve(association.track_ids.size());
  for (std::size_t index = 0; index < association.track_ids.size(); ++index) {
    wire_association.selections.emplace_back(association.track_ids[index],
                                             association.selected_cue_indexes[index]);
  }
  const auto live_ids = source.association->live_ids();
  wire_association.live_track_ids.assign(live_ids.begin(), live_ids.end());
  payload.association = std::move(wire_association);
  return payload;
}

}  // namespace

void publish_metadata(ServerState& state, const ipc::Message& request, SourceSlot& source,
                      const trt::PerceptionResult* result, std::uint64_t source_time_ns) {
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
  metadata.payload = result == nullptr
                         ? encode_empty_perception(metadata)
                         : encode_perception(
                               metadata,
                               perception_payload(state, source, *result, source_time_ns));
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
                           const DecodedFrameView& view) {
  // Pace before inference. Decode and the AU tee remain source-rate; only the
  // expensive perception branch is admitted at the configured policy rate.
  {
    std::lock_guard lock{slot_mutex};
    const auto found = sources.find(camera);
    if (found == sources.end() || found->second.binding != binding) return;
    const std::uint64_t interval_ns = 1'000'000'000ULL / options.target_fps;
    const std::uint64_t last_admit = found->second.last_inference_source_time_ns;
    // Diagnostic: name why a frame was skipped or admitted, so an effective
    // rate below the configured target can be attributed instead of guessed
    // at. Sampled so a 30fps source cannot flood the log.
    const auto log_pace = [&](const char* verb) {
      std::fprintf(stderr,
                   "seeon-pace: camera=%s %s target_fps=%u interval_ns=%llu "
                   "gap_ns=%lld skips=%llu admits=%llu\n",
                   camera.c_str(), verb, options.target_fps,
                   static_cast<unsigned long long>(interval_ns),
                   last_admit == 0 ? -1LL
                                   : static_cast<long long>(view.source_time_ns) -
                                         static_cast<long long>(last_admit),
                   static_cast<unsigned long long>(found->second.pace_skips),
                   static_cast<unsigned long long>(found->second.pace_admits));
    };
    if (last_admit != 0 && view.source_time_ns < last_admit + interval_ns) {
      if ((++found->second.pace_skips % 128) == 0) log_pace("skip");
      return;
    }
    if ((++found->second.pace_admits % 64) == 0) log_pace("admit");
    found->second.last_inference_source_time_ns = view.source_time_ns;
  }
  // Inference runs outside the slot lock: a binding that rolls mid-inference
  // simply drops the stale result at dispatch.
  trt::PerceptionResult result;
  const trt::PerceptionResult* published = nullptr;
  if (perception != nullptr && view.rgba != nullptr) {
    std::string error;
    if (!perception->infer(view.rgba, view.width, view.height, view.stride,
                           options.person_box_from_person_engine, &result, &error)) {
      on_failure({camera, "tensorrt", FailureScope::kFatal});
      return;
    }
    published = &result;
  }
  static_cast<void>(binding->dispatch_frame(
      [this, &camera, &view, &binding, published](std::uint32_t generation,
                                                  std::uint64_t epoch) {
        std::lock_guard lock{slot_mutex};
        const auto found = sources.find(camera);
        if (found == sources.end() || found->second.binding != binding) return;
        ipc::Message message{};
        message.header.worker_boot_id = options.worker_boot_id;
        message.header.child_instance_id = options.child_instance_id;
        message.header.source_generation = generation;
        message.header.stream_epoch = epoch;
        message.header.source_pts = view.pts;
        message.camera = camera;
        message.transform = "seeon-perception-v1";
        command_support::publish_metadata(*this, message, found->second, published,
                                          view.source_time_ns);
      }));
}
}  // namespace seeon
