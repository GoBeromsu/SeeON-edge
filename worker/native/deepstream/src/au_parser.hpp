#pragma once

#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace seeon {
struct AnnexbFacts {
  std::optional<bool> keyframe;
  std::vector<std::uint8_t> codec_data;
};

[[nodiscard]] AnnexbFacts parse_annexb(const std::uint8_t* data, std::size_t size, bool h265);
[[nodiscard]] std::optional<bool> parse_length_prefixed_keyframe(
    const std::uint8_t* data, std::size_t size, bool h265, std::size_t length_size);
}  // namespace seeon
