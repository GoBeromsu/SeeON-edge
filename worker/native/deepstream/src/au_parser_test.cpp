#include "au_parser.hpp"

#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

namespace {
void check(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::abort();
  }
}
}  // namespace

int main() {
  const std::vector<std::uint8_t> h264{
      0, 0, 0, 1, 0x67, 0x64, 0, 0x1f,
      0, 0, 0, 1, 0x68, 0xee, 0x3c, 0x80,
      0, 0, 0, 1, 0x65, 0x80};
  const auto annexb = seeon::parse_annexb(h264.data(), h264.size(), false);
  check(annexb.keyframe == true, "H.264 IDR was not identified");
  check(annexb.codec_data.size() == 16, "H.264 SPS/PPS codec data was not retained");

  const std::vector<std::uint8_t> h265{0, 0, 0, 3, 0x26, 0x01, 0x80};
  check(seeon::parse_length_prefixed_keyframe(h265.data(), h265.size(), true, 4) == true,
        "H.265 IRAP was not identified");
  check(!seeon::parse_length_prefixed_keyframe(h265.data(), h265.size(), true, 0).has_value(),
        "unknown NAL framing did not fail closed");

  std::vector<std::uint8_t> excessive;
  for (std::size_t index = 0; index < 4097; ++index) {
    excessive.insert(excessive.end(), {0, 0, 1, 0x41});
  }
  check(!seeon::parse_annexb(excessive.data(), excessive.size(), false).keyframe.has_value(),
        "excessive Annex-B NAL count did not fail closed");
  return 0;
}
