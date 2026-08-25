// Cross-language parity driver for native_perception.{hpp,cpp}.
//
// Reads a line protocol on stdin and answers on stdout so the Python parity
// suite (tests/test_native_perception_cpp_parity.py) can compare this exact
// production code against the C3/C4 Python references. All floating-point
// values cross the boundary as C99 hexfloats ("%a"), never decimal, so the
// comparison is bit-exact. Prototype tensors are synthesized on both sides
// from the same SplitMix64 stream to keep fixtures small.

#include "native_perception.hpp"

#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace {

using seeon::perception::AffineMetadata;
using seeon::perception::LegacyGreedyBboxIou;
using seeon::perception::ParsedBox;

double parse_hex(const std::string& token) { return std::strtod(token.c_str(), nullptr); }

std::string hex(double value) {
  char buffer[64];
  std::snprintf(buffer, sizeof(buffer), "%a", value);
  return buffer;
}

std::vector<double> read_rows(std::istream& input, int count, int stride) {
  std::vector<double> rows;
  rows.reserve(static_cast<std::size_t>(count) * static_cast<std::size_t>(stride));
  for (int row = 0; row < count; ++row) {
    std::string line;
    std::getline(input, line);
    std::istringstream tokens{line};
    std::string token;
    while (tokens >> token) rows.push_back(parse_hex(token));
  }
  return rows;
}

// Deterministic cross-language PRNG: SplitMix64 -> double in [-1, 1).
class SplitMix64 {
 public:
  explicit SplitMix64(std::uint64_t seed) : state_(seed) {}
  double next_signed_unit() {
    state_ += 0x9E3779B97F4A7C15ULL;
    std::uint64_t z = state_;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
    z = z ^ (z >> 31);
    const double unit = static_cast<double>(z >> 11) * 0x1.0p-53;
    return unit * 2.0 - 1.0;
  }

 private:
  std::uint64_t state_;
};

std::vector<double> synthesize_prototypes(std::uint64_t seed) {
  constexpr std::size_t kSize = 32ULL * 160ULL * 160ULL;
  SplitMix64 generator{seed};
  std::vector<double> values;
  values.reserve(kSize);
  for (std::size_t index = 0; index < kSize; ++index) {
    values.push_back(generator.next_signed_unit());
  }
  return values;
}

void print_affine(const AffineMetadata& affine) {
  std::printf("AFFINE %d %d %d %d %d %d %s %d %d %s %s\n", affine.source_height,
              affine.source_width, affine.tensor_height, affine.tensor_width,
              affine.content_height, affine.content_width, hex(affine.gain).c_str(),
              affine.box_pad_x, affine.box_pad_y, hex(affine.keypoint_pad_x).c_str(),
              hex(affine.keypoint_pad_y).c_str());
}

int run() {
  std::string line;
  std::unique_ptr<LegacyGreedyBboxIou> association;
  while (std::getline(std::cin, line)) {
    std::istringstream tokens{line};
    std::string command;
    tokens >> command;
    if (command == "AFFINE") {
      int height = 0;
      int width = 0;
      tokens >> height >> width;
      print_affine(seeon::perception::letterbox_affine(height, width));
    } else if (command == "INVBOX") {
      int height = 0;
      int width = 0;
      std::string x1, y1, x2, y2;
      tokens >> height >> width >> x1 >> y1 >> x2 >> y2;
      const auto affine = seeon::perception::letterbox_affine(height, width);
      const auto box = affine.invert_box(parse_hex(x1), parse_hex(y1), parse_hex(x2),
                                         parse_hex(y2));
      std::printf("BOX %d %d %d %d\n", box.x1, box.y1, box.x2, box.y2);
    } else if (command == "INVKP") {
      int height = 0;
      int width = 0;
      std::string x, y;
      tokens >> height >> width >> x >> y;
      const auto affine = seeon::perception::letterbox_affine(height, width);
      const auto [sx, sy] = affine.invert_keypoint(parse_hex(x), parse_hex(y));
      std::printf("KP %d %d\n", sx, sy);
    } else if (command == "POSE") {
      int height = 0;
      int width = 0;
      int count = 0;
      tokens >> height >> width >> count;
      const auto rows = read_rows(std::cin, count, seeon::perception::kPoseRowStride);
      const auto affine = seeon::perception::letterbox_affine(height, width);
      const auto parsed = seeon::perception::parse_pose_rows(rows, affine);
      std::printf("POSECOUNT %zu\n", parsed.boxes.size());
      for (std::size_t index = 0; index < parsed.boxes.size(); ++index) {
        const ParsedBox& box = parsed.boxes[index];
        std::printf("POSEBOX %d %d %d %d %s\n", box.x1, box.y1, box.x2, box.y2,
                    hex(box.confidence).c_str());
        std::printf("POSEKP");
        for (const auto& point : parsed.poses[index]) {
          std::printf(" %d %d %s", point.x, point.y, hex(point.score).c_str());
        }
        std::printf("\n");
      }
    } else if (command == "PERSON") {
      int height = 0;
      int width = 0;
      std::string confidence;
      int count = 0;
      tokens >> height >> width >> confidence >> count;
      const auto rows = read_rows(std::cin, count, seeon::perception::kPersonRowStride);
      const auto affine = seeon::perception::letterbox_affine(height, width);
      const auto boxes =
          seeon::perception::parse_person_rows(rows, affine, parse_hex(confidence));
      std::printf("PERSONCOUNT %zu\n", boxes.size());
      for (const ParsedBox& box : boxes) {
        std::printf("PERSONBOX %d %d %d %d %s\n", box.x1, box.y1, box.x2, box.y2,
                    hex(box.confidence).c_str());
      }
    } else if (command == "BED") {
      int height = 0;
      int width = 0;
      std::string confidence;
      std::uint64_t seed = 0;
      int max_points = 0;
      int count = 0;
      tokens >> height >> width >> confidence >> seed >> max_points >> count;
      const auto rows = read_rows(std::cin, count, seeon::perception::kBedRowStride);
      const auto prototypes = synthesize_prototypes(seed);
      const auto affine = seeon::perception::letterbox_affine(height, width);
      const auto regions = seeon::perception::parse_bed_rows(
          rows, prototypes, affine, parse_hex(confidence), max_points);
      std::printf("BEDCOUNT %zu\n", regions.size());
      for (const auto& region : regions) {
        std::printf("BEDBOX %d %d %d %d %s\n", region.bounds.x1, region.bounds.y1,
                    region.bounds.x2, region.bounds.y2, hex(region.bounds.confidence).c_str());
        std::printf("BEDPOLY %zu", region.polygon.size());
        for (const auto& [x, y] : region.polygon) std::printf(" %d %d", x, y);
        std::printf("\n");
      }
    } else if (command == "ASSOC") {
      std::string action;
      tokens >> action;
      if (action == "NEW") {
        std::string min_iou;
        int max_misses = 0;
        tokens >> min_iou >> max_misses;
        association = std::make_unique<LegacyGreedyBboxIou>(parse_hex(min_iou), max_misses);
        std::printf("ACK\n");
      } else if (action == "OBSERVE") {
        int count = 0;
        tokens >> count;
        std::vector<ParsedBox> boxes;
        boxes.reserve(static_cast<std::size_t>(count));
        for (int index = 0; index < count; ++index) {
          std::string row;
          std::getline(std::cin, row);
          std::istringstream fields{row};
          ParsedBox box{};
          std::string conf;
          fields >> box.x1 >> box.y1 >> box.x2 >> box.y2 >> conf;
          box.confidence = parse_hex(conf);
          boxes.push_back(box);
        }
        const auto output = association->observe(boxes);
        std::printf("ASSOC %zu", output.track_ids.size());
        for (std::size_t index = 0; index < output.track_ids.size(); ++index) {
          std::printf(" %" PRId64 ":%u", output.track_ids[index],
                      output.selected_cue_indexes[index]);
        }
        std::printf("\n");
      } else if (action == "COAST") {
        association->coast();
        std::printf("ACK\n");
      } else if (action == "RESET") {
        association->reset();
        std::printf("ACK\n");
      } else {
        std::printf("ERR unknown-assoc-action\n");
        return 2;
      }
    } else if (command.empty()) {
      continue;
    } else {
      std::printf("ERR unknown-command\n");
      return 2;
    }
    std::fflush(stdout);
  }
  return 0;
}

}  // namespace

int main() { return run(); }
