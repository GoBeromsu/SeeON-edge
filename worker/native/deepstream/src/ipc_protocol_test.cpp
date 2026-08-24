#include "ipc_protocol.hpp"
#include "perception_wire.hpp"

#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
void check(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::abort();
  }
}

std::string hex(const std::vector<std::uint8_t>& payload) {
  std::ostringstream output;
  for (const auto value : payload) {
    output << std::hex << std::setw(2) << std::setfill('0') << static_cast<int>(value);
  }
  return output.str();
}
}  // namespace

int main(int argc, char** argv) {
  seeon::ipc::Message message{};
  message.header.kind = static_cast<std::uint8_t>(seeon::ipc::Kind::kAddSource);
  message.header.source_generation = 3;
  message.header.stream_epoch = 7;
  message.header.source_pts = 123456;
  message.header.source_sequence = 11;
  message.header.native_publish_sequence = 13;
  message.header.request_id = 14;
  message.header.worker_boot_id = {0x12, 0x34, 0x56, 0x78, 0x12, 0x34, 0x56, 0x78,
                                   0x12, 0x34, 0x56, 0x78, 0x12, 0x34, 0x56, 0x78};
  message.camera = "camera-a";
  message.transform = "seeon-perception-v1";
  message.payload = {1, 2, 3};

  const std::vector<std::uint8_t> encoded = seeon::ipc::encode(message);
  const auto decoded = seeon::ipc::decode(encoded);
  check(decoded.has_value(), "IPC round-trip did not decode");
  check(decoded->header.source_generation == 3, "source generation mismatch");
  check(decoded->header.stream_epoch == 7, "stream epoch mismatch");
  check(decoded->header.source_pts == 123456, "source PTS mismatch");
  check(decoded->header.source_sequence == 11, "source sequence mismatch");
  check(decoded->header.native_publish_sequence == 13, "publish sequence mismatch");
  check(decoded->header.request_id == 14, "request ID mismatch");
  check(decoded->camera == "camera-a", "camera identity mismatch");
  check(decoded->transform == "seeon-perception-v1", "transform identity mismatch");
  check(decoded->payload == std::vector<std::uint8_t>({1, 2, 3}), "payload mismatch");

  std::vector<std::uint8_t> malformed = encoded;
  malformed[0] = 0;
  check(!seeon::ipc::decode(malformed).has_value(), "malformed IPC frame was accepted");

  check(hex(seeon::encode_empty_perception(message)) ==
            "5046563112345678123456781234567812345678080063616d6572612d61"
            "070000000000000040e20100000000000b0000000000000002020200000000000000",
        "empty Python/C++ golden vector mismatch");

  seeon::PerceptionPayload frame;
  frame.person_state = 1;
  frame.pose_state = 1;
  frame.bed_state = 1;
  frame.boxes = {{1, 2, 30, 40, 0.75}, {5, 6, 50, 60, 0.5}};
  frame.poses = {{{3, 4, 0.9}, {7, 8, 0.8}}, {{10, 11, 0.7}}};
  frame.bed_regions = {{{0, 0, 100, 80, 0.95}, {{0, 0}, {100, 0}, {100, 80}}}};
  frame.association = seeon::WireAssociation{
      "legacy-greedy-bbox-iou.v1", "person_box", {{41, 0}, {42, 1}}};
  const std::string nonempty = hex(seeon::encode_perception(message, frame));
  if (argc == 2 && std::string{argv[1]} == "--emit-nonempty") {
    std::cout << nonempty << '\n';
    return 0;
  }
  check(nonempty ==
            "5046563112345678123456781234567812345678080063616d6572612d61"
            "070000000000000040e20100000000000b0000000000000001010101020001000000"
            "020000001e00000028000000000000000000e83f0500000006000000320000003c00"
            "0000000000000000e03f020002000300000004000000cdccccccccccec3f07000000"
            "080000009a9999999999e93f01000a0000000b000000666666666666e63f01000000"
            "0000000000006400000050000000666666666666ee3f030000000000000000006400"
            "00000000000064000000500000001234567812345678123456781234567808006361"
            "6d6572612d61070000000000000040e20100000000000b0000000000000019006c65"
            "676163792d6772656564792d62626f782d696f752e76310a00706572736f6e5f626f"
            "780200290000000000000000002a000000000000000100",
        "non-empty Python/C++ golden vector mismatch");

  frame.person_state = 2;
  bool rejected = false;
  try {
    (void)seeon::encode_perception(message, frame);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  check(rejected, "inferred-empty channel with boxes was accepted");
  frame.person_state = 1;
  frame.association->cue_source = "bed_region";
  rejected = false;
  try {
    (void)seeon::encode_perception(message, frame);
  } catch (const std::runtime_error&) {
    rejected = true;
  }
  check(rejected, "bed-region identity cue was accepted");
  return 0;
}
