#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace seeon {
struct ChildOptions {
  int control_fd = -1;
  int wake_fd = -1;
  int au_fd = -1;
  int preview_fd = -1;
  int failure_fd = -1;
  int ready_fd = -1;
  std::uint32_t parent_pid = 0;
  bool qa_mode = false;
  // "pose" (default) mirrors the shipped box_source: person_box cues come from
  // the pose head and the person engine is not run per frame.
  bool person_box_from_person_engine = false;
  // Content-addressed engine cache directory; "-" disables perception (QA
  // lifecycle runs only; production spawn always passes a real path).
  std::string engine_cache;
  std::uint32_t target_fps = 15;
  std::array<std::uint8_t, 16> worker_boot_id;
  std::array<std::uint8_t, 16> child_instance_id;
};

class ChildServer {
 public:
  explicit ChildServer(ChildOptions options);
  [[nodiscard]] int run();

 private:
  ChildOptions options_;
};
}  // namespace seeon
