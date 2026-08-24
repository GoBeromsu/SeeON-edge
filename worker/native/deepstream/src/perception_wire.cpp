#include "perception_wire.hpp"

#include <cstring>
#include <limits>
#include <stdexcept>

namespace seeon {
namespace {
constexpr std::size_t kMaximumItems = 256;
constexpr std::size_t kMaximumKeypoints = 64;
constexpr std::size_t kMaximumText = 128;

template <typename Value>
void append(std::vector<std::uint8_t>& output, Value value) {
  const auto offset = output.size();
  output.resize(offset + sizeof(value));
  std::memcpy(output.data() + offset, &value, sizeof(value));
}

void append_text(std::vector<std::uint8_t>& output, const std::string& value) {
  if (value.size() > kMaximumText) throw std::runtime_error{"wire text too large"};
  append(output, static_cast<std::uint16_t>(value.size()));
  output.insert(output.end(), value.begin(), value.end());
}

void append_identity(std::vector<std::uint8_t>& output, const ipc::Message& envelope) {
  if (envelope.camera.size() > kMaximumText) {
    throw std::runtime_error{"camera identity too large"};
  }
  output.insert(output.end(), envelope.header.worker_boot_id.begin(),
                envelope.header.worker_boot_id.end());
  append_text(output, envelope.camera);
  append(output, envelope.header.stream_epoch);
  append(output, envelope.header.source_pts);
  append(output, envelope.header.source_sequence);
}

void append_box(std::vector<std::uint8_t>& output, const WireBox& box) {
  append(output, box.x1);
  append(output, box.y1);
  append(output, box.x2);
  append(output, box.y2);
  append(output, box.confidence);
}

void append_count(std::vector<std::uint8_t>& output, std::size_t count) {
  if (count > kMaximumItems || count > std::numeric_limits<std::uint16_t>::max()) {
    throw std::runtime_error{"wire item count too large"};
  }
  append(output, static_cast<std::uint16_t>(count));
}

bool valid_state(std::uint8_t state) { return state >= 1 && state <= 3; }

bool valid_channel(std::uint8_t state, std::size_t count) {
  return state == 1 ? count != 0 : count == 0;
}
}  // namespace

std::vector<std::uint8_t> encode_perception(const ipc::Message& envelope,
                                            const PerceptionPayload& frame) {
  if (!valid_state(frame.person_state) || !valid_state(frame.pose_state) ||
      !valid_state(frame.bed_state) ||
      !valid_channel(frame.person_state, frame.boxes.size()) ||
      !valid_channel(frame.pose_state, frame.poses.size()) ||
      !valid_channel(frame.bed_state, frame.bed_regions.size())) {
    throw std::runtime_error{"invalid channel state"};
  }
  std::vector<std::uint8_t> payload{'P', 'F', 'V', '1'};
  append_identity(payload, envelope);
  payload.insert(payload.end(), {frame.person_state, frame.pose_state, frame.bed_state,
                                 static_cast<std::uint8_t>(frame.association.has_value())});
  append_count(payload, frame.boxes.size());
  for (const WireBox& box : frame.boxes) append_box(payload, box);
  append_count(payload, frame.poses.size());
  for (const auto& pose : frame.poses) {
    if (pose.size() > kMaximumKeypoints) throw std::runtime_error{"too many keypoints"};
    append_count(payload, pose.size());
    for (const WireKeypoint& point : pose) {
      append(payload, point.x);
      append(payload, point.y);
      append(payload, point.score);
    }
  }
  append_count(payload, frame.bed_regions.size());
  for (const WireBedRegion& region : frame.bed_regions) {
    append_box(payload, region.bounds);
    append_count(payload, region.polygon.size());
    for (const auto& [x, y] : region.polygon) {
      append(payload, x);
      append(payload, y);
    }
  }
  if (frame.association.has_value()) {
    const WireAssociation& association = *frame.association;
    if (association.cue_source != "person_box") {
      throw std::runtime_error{"invalid association cue source"};
    }
    append_identity(payload, envelope);
    append_text(payload, association.strategy);
    append_text(payload, association.cue_source);
    append_count(payload, association.selections.size());
    for (const auto& [track_id, cue_index] : association.selections) {
      if (cue_index >= frame.boxes.size()) {
        throw std::runtime_error{"association cue index out of range"};
      }
      append(payload, track_id);
      append(payload, cue_index);
    }
  }
  return payload;
}

std::vector<std::uint8_t> encode_empty_perception(const ipc::Message& envelope) {
  return encode_perception(envelope, PerceptionPayload{});
}
}  // namespace seeon
