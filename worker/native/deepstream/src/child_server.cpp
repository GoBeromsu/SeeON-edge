#include "child_server.hpp"

#include "child_commands.hpp"
#include "ipc_protocol.hpp"

#include <poll.h>
#include <signal.h>
#include <sys/prctl.h>
#include <sys/signalfd.h>
#include <sys/socket.h>
#include <unistd.h>

#include <utility>
#include <vector>

namespace seeon {
ChildServer::ChildServer(ChildOptions options) : options_(std::move(options)) {}

int ChildServer::run() {
  if (prctl(PR_SET_PDEATHSIG, SIGKILL) < 0 ||
      static_cast<std::uint32_t>(getppid()) != options_.parent_pid ||
      (getppid() == 1 && options_.parent_pid != 1)) {
    return 4;
  }
  sigset_t signals;
  sigemptyset(&signals);
  sigaddset(&signals, SIGTERM);
  sigaddset(&signals, SIGINT);
  pthread_sigmask(SIG_BLOCK, &signals, nullptr);
  const int signal_fd = signalfd(-1, &signals, SFD_CLOEXEC);
  if (signal_fd < 0) {
    return 4;
  }
  ServerState state{options_};
  if (!state.runtime.custom_transform_available()) {
    close(signal_fd);
    return 4;
  }
  const auto ready_written = write(options_.ready_fd, "R", 1);
  close(options_.ready_fd);
  if (ready_written != 1) {
    close(signal_fd);
    return 4;
  }
  int exit_code = 4;
  std::vector<std::uint8_t> buffer(65'535);
  while (true) {
    pollfd active[] = {
        {options_.control_fd, POLLIN, 0},
        {signal_fd, POLLIN, 0},
        {state.failure_fd, POLLIN, 0},
    };
    if (poll(active, 3, -1) < 0) {
      break;
    }
    if (active[1].revents != 0) {
      exit_code = 0;
      break;
    }
    if (active[2].revents != 0) {
      const int failure_exit = handle_runtime_failures(state);
      if (failure_exit >= 0) {
        exit_code = failure_exit;
        break;
      }
      continue;
    }
    if ((active[0].revents & (POLLHUP | POLLERR | POLLNVAL)) != 0) {
      break;
    }
    if ((active[0].revents & POLLIN) == 0) {
      continue;
    }
    const auto received = recv(options_.control_fd, buffer.data(), buffer.size(), 0);
    if (received <= 0) {
      break;
    }
    std::vector<std::uint8_t> frame(buffer.begin(), buffer.begin() + received);
    const auto request = ipc::decode(frame);
    if (!request.has_value()) {
      ++state.malformed;
      break;
    }
    CommandResult result = handle_command(state, *request);
    const auto reply = ipc::encode(result.reply);
    const auto sent = send(options_.control_fd, reply.data(), reply.size(), MSG_NOSIGNAL);
    if (sent != static_cast<ssize_t>(reply.size())) {
      break;
    }
    if (result.exit_code >= 0) {
      exit_code = result.exit_code;
      break;
    }
  }
  close(options_.control_fd);
  close(options_.wake_fd);
  close(options_.failure_fd);
  close(signal_fd);
  return exit_code;
}
}  // namespace seeon
