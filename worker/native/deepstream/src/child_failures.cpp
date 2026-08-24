#include "child_commands.hpp"

#include <sys/eventfd.h>
#include <sys/socket.h>
#include <unistd.h>

#include <string>
#include <vector>

namespace seeon {
ServerState::ServerState(const ChildOptions& options_value)
    : options(options_value),
      runtime(
          [this](const std::string& camera, std::uint64_t pts) { on_frame(camera, pts); },
          [this](const NativeFailure& failure) { on_failure(failure); }),
      failure_fd(eventfd(0, EFD_CLOEXEC | EFD_NONBLOCK)) {}

ServerState::~ServerState() { close(failure_fd); }

void ServerState::on_failure(const NativeFailure& failure) {
  {
    constexpr std::size_t kFailureCapacity = 64;
    std::lock_guard lock{failure_mutex};
    if (failures.size() >= kFailureCapacity) {
      failures.clear();
      failures.push_back({"_worker", "failure_overflow", FailureScope::kFatal});
    } else {
      failures.push_back(failure);
    }
  }
  static_cast<void>(eventfd_write(failure_fd, 1));
}

std::deque<NativeFailure> ServerState::take_failures() {
  eventfd_t ignored = 0;
  static_cast<void>(eventfd_read(failure_fd, &ignored));
  std::lock_guard lock{failure_mutex};
  std::deque<NativeFailure> result;
  result.swap(failures);
  return result;
}

int handle_runtime_failures(ServerState& state) {
  for (const NativeFailure& failure : state.take_failures()) {
    ipc::Message event{};
    event.header.kind = static_cast<std::uint8_t>(
        failure.scope == FailureScope::kFatal ? ipc::Kind::kFatal : ipc::Kind::kSourceFailure);
    event.header.worker_boot_id = state.options.worker_boot_id;
    event.header.child_instance_id = state.options.child_instance_id;
    event.camera = failure.camera.empty() ? "_worker" : failure.camera;
    event.transform = "seeon-perception-v1";
    event.payload = {failure.category.begin(), failure.category.end()};
    const std::vector<std::uint8_t> encoded = ipc::encode(event);
    const auto sent =
        send(state.options.failure_fd, encoded.data(), encoded.size(), MSG_NOSIGNAL);
    if (sent != static_cast<ssize_t>(encoded.size())) {
      return 4;
    }
    if (failure.scope == FailureScope::kFatal) {
      return 4;
    }
    std::lock_guard lock{state.slot_mutex};
    ++state.source_failures;
  }
  return -1;
}
}  // namespace seeon
