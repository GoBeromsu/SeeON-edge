#include "child_commands.hpp"

#include <cstdlib>
#include <iostream>
#include <mutex>
#include <string>

namespace {
void check(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::abort();
  }
}

seeon::ipc::Message request(const seeon::ChildOptions& options, seeon::ipc::Kind kind,
                            std::uint32_t generation, std::string payload = {}) {
  seeon::ipc::Message message{};
  message.header.kind = static_cast<std::uint8_t>(kind);
  message.header.worker_boot_id = options.worker_boot_id;
  message.header.child_instance_id = options.child_instance_id;
  message.header.source_generation = generation;
  message.camera = "camera-a";
  message.transform = "seeon-perception-v1";
  message.payload = {payload.begin(), payload.end()};
  return message;
}
}  // namespace

int main() {
  seeon::ChildOptions options{};
  options.worker_boot_id[0] = 1;
  options.child_instance_id[0] = 2;
  options.qa_mode = false;
  seeon::ServerState state{options};

  auto add = request(options, seeon::ipc::Kind::kAddSource, 1, "loopback://camera-a");
  auto result = seeon::handle_command(state, add);
  check(result.reply.header.kind == static_cast<std::uint8_t>(seeon::ipc::Kind::kEpochStarted),
        "initial source add failed");
  check(result.reply.header.stream_epoch == 1, "initial epoch mismatch");

  const seeon::ipc::Kind qa_commands[] = {
      seeon::ipc::Kind::kEmitMetadata,
      seeon::ipc::Kind::kWaitPublish,
      seeon::ipc::Kind::kInjectSourceEos,
  };
  for (const auto command : qa_commands) {
    result = seeon::handle_command(state, request(options, command, 1));
    check(result.reply.header.kind == static_cast<std::uint8_t>(seeon::ipc::Kind::kError),
          "QA command was accepted outside QA mode");
    check(std::string{result.reply.payload.begin(), result.reply.payload.end()} ==
              "qa_command_disabled",
          "QA command refusal was not explicit");
  }

  auto remove = request(options, seeon::ipc::Kind::kRemoveSource, 1);
  result = seeon::handle_command(state, remove);
  check(result.reply.header.kind == static_cast<std::uint8_t>(seeon::ipc::Kind::kAck),
        "source remove failed");

  add = request(options, seeon::ipc::Kind::kAddSource, 2, "loopback://camera-a");
  result = seeon::handle_command(state, add);
  check(result.reply.header.stream_epoch == 1, "re-add did not start a fresh epoch");

  seeon::PipelineBindingPtr retired_binding;
  {
    std::lock_guard lock{state.slot_mutex};
    retired_binding = state.sources.at("camera-a").binding;
  }
  auto rebuild = request(options, seeon::ipc::Kind::kSourceFailure, 2, "eos");
  result = seeon::handle_command(state, rebuild);
  check(result.reply.header.stream_epoch == 2, "reconnect did not advance epoch");
  {
    std::lock_guard lock{state.slot_mutex};
    const auto found = state.sources.find("camera-a");
    check(found != state.sources.end() && found->second.latest.has_value(),
          "replacement pipeline did not publish an early frame");
    check(found->second.latest->header.stream_epoch == 2,
          "replacement pipeline early frame carried the old epoch");
    check(found->second.binding != retired_binding,
          "replacement pipeline retained the old pipeline token");
  }
  bool old_pipeline_dispatched = false;
  check(!retired_binding->dispatch_au(
            [&old_pipeline_dispatched](std::uint32_t, std::uint64_t, std::uint64_t) {
              old_pipeline_dispatched = true;
            }),
        "retired pipeline token remained live after epoch roll");
  check(!old_pipeline_dispatched, "retired pipeline emitted a cross-epoch AU");

  result = seeon::handle_command(state, remove);
  check(result.reply.header.kind == static_cast<std::uint8_t>(seeon::ipc::Kind::kAck),
        "final source remove failed");
  return 0;
}
