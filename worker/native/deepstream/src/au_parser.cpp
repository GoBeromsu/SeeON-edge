#include "au_parser.hpp"

#include <utility>

namespace seeon {
namespace {
constexpr std::size_t kMaxNalUnits = 4096;
struct NalUnit {
  std::size_t prefix;
  std::size_t payload;
  std::size_t end;
};

std::vector<NalUnit> annexb_units(const std::uint8_t* data, std::size_t size) {
  std::vector<std::pair<std::size_t, std::size_t>> starts;
  for (std::size_t index = 0; index + 3 < size; ++index) {
    if (data[index] != 0 || data[index + 1] != 0) continue;
    if (data[index + 2] == 1) {
      starts.emplace_back(index, 3);
      if (starts.size() > kMaxNalUnits) return {};
      index += 2;
    } else if (index + 4 < size && data[index + 2] == 0 && data[index + 3] == 1) {
      starts.emplace_back(index, 4);
      if (starts.size() > kMaxNalUnits) return {};
      index += 3;
    }
  }
  std::vector<NalUnit> units;
  for (std::size_t index = 0; index < starts.size(); ++index) {
    const auto [offset, prefix_size] = starts[index];
    const auto end = index + 1 < starts.size() ? starts[index + 1].first : size;
    if (offset + prefix_size < end) units.push_back({offset, offset + prefix_size, end});
  }
  return units;
}

unsigned int nal_type(const std::uint8_t* data, bool h265) {
  return h265 ? static_cast<unsigned int>((data[0] >> 1U) & 0x3fU)
              : static_cast<unsigned int>(data[0] & 0x1fU);
}

bool parameter_set(unsigned int type, bool h265) {
  return h265 ? type >= 32U && type <= 34U : type == 7U || type == 8U;
}
}  // namespace

AnnexbFacts parse_annexb(const std::uint8_t* data, std::size_t size, bool h265) {
  AnnexbFacts facts{};
  for (const auto& unit : annexb_units(data, size)) {
    const auto type = nal_type(data + unit.payload, h265);
    if (parameter_set(type, h265)) {
      facts.codec_data.insert(facts.codec_data.end(), data + unit.prefix, data + unit.end);
    }
    if ((!h265 && type == 5U) || (h265 && type >= 16U && type <= 23U)) {
      facts.keyframe = true;
    } else if (!facts.keyframe.has_value() &&
               ((!h265 && type >= 1U && type <= 5U) || (h265 && type <= 31U))) {
      facts.keyframe = false;
    }
  }
  return facts;
}

std::optional<bool> parse_length_prefixed_keyframe(const std::uint8_t* data, std::size_t size,
                                                    bool h265, std::size_t length_size) {
  if (length_size == 0 || length_size > 4) return std::nullopt;
  std::size_t offset = 0;
  std::optional<bool> keyframe;
  while (offset + length_size < size) {
    std::size_t nal_size = 0;
    for (std::size_t index = 0; index < length_size; ++index) {
      nal_size = (nal_size << 8U) | data[offset + index];
    }
    offset += length_size;
    if (nal_size == 0 || offset + nal_size > size) return std::nullopt;
    const auto type = nal_type(data + offset, h265);
    if ((!h265 && type == 5U) || (h265 && type >= 16U && type <= 23U)) {
      keyframe = true;
    } else if (!keyframe.has_value() &&
               ((!h265 && type >= 1U && type <= 5U) || (h265 && type <= 31U))) {
      keyframe = false;
    }
    offset += nal_size;
  }
  return keyframe;
}
}  // namespace seeon
