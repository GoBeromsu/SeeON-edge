#pragma once

#include "ipc_protocol.hpp"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

namespace seeon {
struct WireBox {
  std::int32_t x1;
  std::int32_t y1;
  std::int32_t x2;
  std::int32_t y2;
  double confidence;
};
struct WireKeypoint {
  std::int32_t x;
  std::int32_t y;
  double score;
};
struct WireBedRegion {
  WireBox bounds;
  std::vector<std::pair<std::int32_t, std::int32_t>> polygon;
};
struct WireAssociation {
  std::string strategy;
  std::string cue_source;
  std::vector<std::pair<std::int64_t, std::uint16_t>> selections;
  std::vector<std::int64_t> live_track_ids;
};
struct PerceptionPayload {
  std::uint8_t person_state = 2;
  std::uint8_t pose_state = 2;
  std::uint8_t bed_state = 2;
  // Wire-v2 geometry/source-time carrier (0 == not provided).
  std::uint16_t source_width = 0;
  std::uint16_t source_height = 0;
  std::uint64_t source_time_ns = 0;
  std::vector<WireBox> boxes;
  std::vector<std::vector<WireKeypoint>> poses;
  std::vector<WireBedRegion> bed_regions;
  std::optional<WireAssociation> association;
};

[[nodiscard]] std::vector<std::uint8_t> encode_perception(
    const ipc::Message& envelope, const PerceptionPayload& frame);
[[nodiscard]] std::vector<std::uint8_t> encode_empty_perception(const ipc::Message& envelope);
}  // namespace seeon
