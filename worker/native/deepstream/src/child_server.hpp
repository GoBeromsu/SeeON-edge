#pragma once

#include <array>
#include <cstdint>
#include <string>

namespace seeon {
struct ChildOptions {
  int control_fd = -1;
  int wake_fd = -1;
  int failure_fd = -1;
  int ready_fd = -1;
  std::uint32_t parent_pid = 0;
  bool qa_mode = false;
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
