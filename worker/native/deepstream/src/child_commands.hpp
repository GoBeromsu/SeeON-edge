#pragma once

#include "au_transport.hpp"
#include "child_server.hpp"
#include "ipc_protocol.hpp"
#include "native_perception.hpp"
#include "preview_transport.hpp"
#include "source_runtime.hpp"
#include "trt_perception.hpp"

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <string>

namespace seeon {
struct SourceSlot {
  std::uint32_t generation;
  std::uint64_t epoch;
  std::uint64_t source_sequence = 0;
  std::uint64_t last_inference_source_time_ns = 0;
  // Pacing diagnostics: the effective inference rate sits well below the
  // configured target and no resource is saturated, so the admit/skip ratio
  // has to be measurable rather than inferred.
  std::uint64_t pace_skips = 0;
  std::uint64_t pace_admits = 0;
  std::optional<ipc::Message> latest;
  PipelineBindingPtr binding;
  // Per-source association state; a fresh instance on add and on every epoch
  // roll (C4: a rolled epoch can never resume a previous track id).
  std::shared_ptr<perception::LegacyGreedyBboxIou> association;
};

class ServerState {
 public:
  explicit ServerState(const ChildOptions& options_value);
  ~ServerState();

  void on_frame(const std::string& camera, const PipelineBindingPtr& binding,
                const DecodedFrameView& view);
  void on_access_unit(const std::string& camera, const PipelineBindingPtr& binding,
                      ParsedAccessUnit unit);
  void on_failure(const NativeFailure& failure);
  [[nodiscard]] std::deque<NativeFailure> take_failures();

  const ChildOptions& options;
  std::unique_ptr<trt::TrtPerception> perception;
  SourceRuntime runtime;
  AuSender au_sender;
  PreviewSender preview_sender;
  std::mutex slot_mutex;
  std::mutex published_mutex;
  std::condition_variable published_condition;
  int failure_fd;
  std::mutex failure_mutex;
  std::deque<NativeFailure> failures;
  std::map<std::string, SourceSlot> sources;
  std::map<std::string, std::uint32_t> generation_high_water;
  std::uint64_t publish_sequence = 0;
  std::atomic<std::uint64_t> published{0};
  std::uint64_t overwritten = 0;
  std::uint64_t wake_dropped = 0;
  std::uint64_t source_failures = 0;
  std::uint64_t malformed = 0;
};

struct CommandResult {
  ipc::Message reply;
  int exit_code = -1;
};

[[nodiscard]] CommandResult handle_command(ServerState& state, const ipc::Message& request);
[[nodiscard]] int handle_runtime_failures(ServerState& state);
}  // namespace seeon
