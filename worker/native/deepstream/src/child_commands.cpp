#include "child_commands.hpp"

#include "child_command_support.hpp"

#include <chrono>
#include <cstring>
#include <memory>
#include <utility>

namespace seeon {
using command_support::error_reply;
using command_support::identity_matches;
using command_support::publish_metadata;
using command_support::status_reply;
using command_support::text_payload;

CommandResult handle_command(ServerState& state, const ipc::Message& request) {
  if (!identity_matches(state, request)) {
    return {error_reply(request, "identity mismatch"), 4};
  }
  const auto kind = static_cast<ipc::Kind>(request.header.kind);
  const bool debug_command = kind == ipc::Kind::kEmitMetadata ||
                             kind == ipc::Kind::kWaitPublish ||
                             kind == ipc::Kind::kInjectSourceEos ||
                             kind == ipc::Kind::kGetPreviewStatus ||
                             kind == ipc::Kind::kWaitPreview;
  if (debug_command && !state.options.qa_mode) {
    return {error_reply(request, "qa_command_disabled")};
  }
  std::string error;
  switch (kind) {
    case ipc::Kind::kAddSource: {
      const auto binding = std::make_shared<PipelineBinding>(request.header.source_generation, 1);
      {
        std::lock_guard lock{state.slot_mutex};
        if (!state.generation_high_water.contains(request.camera) &&
            state.generation_high_water.size() >= 64) {
          return {error_reply(request, "source_capacity")};
        }
        const auto high_water = state.generation_high_water[request.camera];
        if (request.header.source_generation <= high_water ||
            state.sources.contains(request.camera)) {
          return {error_reply(request, "source generation rejected")};
        }
        state.generation_high_water[request.camera] = request.header.source_generation;
        state.sources.emplace(
            request.camera,
            SourceSlot{request.header.source_generation, 1, 0, std::nullopt, binding});
      }
      const std::string uri{request.payload.begin(), request.payload.end()};
      if (!state.runtime.add(request.camera, uri, binding, &error)) {
        std::lock_guard lock{state.slot_mutex};
        state.sources.erase(request.camera);
        return {error_reply(request, error)};
      }
      ipc::Message reply = ipc::reply(request, ipc::Kind::kEpochStarted);
      reply.header.stream_epoch = 1;
      return {std::move(reply)};
    }
    case ipc::Kind::kRemoveSource: {
      {
        std::lock_guard lock{state.slot_mutex};
        if (!state.sources.contains(request.camera)) {
          return {error_reply(request, "unknown source")};
        }
      }
      if (!state.runtime.remove(request.camera)) {
        return {error_reply(request, "source removal failed")};
      }
      std::lock_guard lock{state.slot_mutex};
      state.sources.erase(request.camera);
      return {ipc::reply(request, ipc::Kind::kAck)};
    }
    case ipc::Kind::kSourceFailure: {
      const std::string category{request.payload.begin(), request.payload.end()};
      const bool source_local = category == "rtsp_dns" || category == "rtsp_connect" ||
          category == "rtsp_auth" || category == "rtsp_timeout" || category == "rtsp_silence" ||
          category == "eos" || category == "depay" || category == "parser" || category == "caps" ||
          category == "decoder_source";
      if (!source_local) {
        return {error_reply(request, "ambiguous source failure"), 4};
      }
      {
        std::lock_guard lock{state.slot_mutex};
        if (!state.sources.contains(request.camera)) {
          return {error_reply(request, "source_unknown")};
        }
      }
      if (!state.runtime.quiesce(request.camera)) {
        return {error_reply(request, "source_quiesce_failed")};
      }
      std::uint64_t epoch = 0;
      PipelineBindingPtr binding;
      {
        std::lock_guard lock{state.slot_mutex};
        const auto found = state.sources.find(request.camera);
        epoch = ++found->second.epoch;
        found->second.latest.reset();
        found->second.source_sequence = 0;
        binding = std::make_shared<PipelineBinding>(found->second.generation, epoch);
        found->second.binding = binding;
      }
      if (!state.runtime.restart(request.camera, binding, &error)) {
        binding->invalidate();
        return {error_reply(request, error)};
      }
      {
        std::lock_guard lock{state.slot_mutex};
        ++state.source_failures;
      }
      ipc::Message reply = ipc::reply(request, ipc::Kind::kEpochStarted);
      reply.header.stream_epoch = epoch;
      return {std::move(reply)};
    }
    case ipc::Kind::kEmitMetadata: {
      std::lock_guard lock{state.slot_mutex};
      const auto found = state.sources.find(request.camera);
      if (found == state.sources.end()) {
        return {error_reply(request, "unknown source")};
      }
      publish_metadata(state, request, found->second);
      return {ipc::reply(request, ipc::Kind::kAck)};
    }
    case ipc::Kind::kGetLatest: {
      std::lock_guard lock{state.slot_mutex};
      const auto found = state.sources.find(request.camera);
      if (found == state.sources.end() || !found->second.latest.has_value()) {
        return {ipc::reply(request, ipc::Kind::kCapabilityInactive)};
      }
      ipc::Message latest = std::move(*found->second.latest);
      found->second.latest.reset();
      latest.header.request_id = request.header.request_id;
      return {std::move(latest)};
    }
    case ipc::Kind::kRecord:
    case ipc::Kind::kSnapshot:
      return {ipc::reply(request, ipc::Kind::kCapabilityInactive,
                         text_payload("dark capability not active"))};
    case ipc::Kind::kStatus: {
      std::lock_guard lock{state.slot_mutex};
      return {status_reply(state, request)};
    }
    case ipc::Kind::kFatal: {
      const std::string category{request.payload.begin(), request.payload.end()};
      const bool fatal = category == "cuda" || category == "xid" || category == "context" ||
                         category == "native_heap" || category == "tensorrt";
      return fatal ? CommandResult{ipc::reply(request, ipc::Kind::kAck), 4}
                   : CommandResult{error_reply(request, "unknown fatal category"), 4};
    }
    case ipc::Kind::kShutdown:
      return {ipc::reply(request, ipc::Kind::kAck), 0};
    case ipc::Kind::kGetSourceState: {
      std::lock_guard lock{state.slot_mutex};
      const auto found = state.sources.find(request.camera);
      if (found == state.sources.end()) {
        return {error_reply(request, "source_unknown")};
      }
      ipc::Message reply = ipc::reply(request, ipc::Kind::kEpochStarted);
      reply.header.source_generation = found->second.generation;
      reply.header.stream_epoch = found->second.epoch;
      return {std::move(reply)};
    }
    case ipc::Kind::kInjectSourceEos:
      return state.runtime.inject_eos(request.camera)
                 ? CommandResult{ipc::reply(request, ipc::Kind::kAck)}
                 : CommandResult{error_reply(request, "source_unknown")};
    case ipc::Kind::kSetPreviewDemand: {
      if (request.payload.size() != sizeof(std::uint32_t)) {
        return {error_reply(request, "preview demand size")};
      }
      std::uint32_t viewers = 0;
      std::memcpy(&viewers, request.payload.data(), sizeof(viewers));
      return state.runtime.set_preview_viewers(request.camera, viewers)
                 ? CommandResult{ipc::reply(request, ipc::Kind::kAck)}
                 : CommandResult{error_reply(request, "source_unknown")};
    }
    case ipc::Kind::kGetPreviewStatus: {
      const auto status = state.runtime.preview_status(request.camera);
      if (!status.has_value()) return {error_reply(request, "source_unknown")};
      std::vector<std::uint8_t> payload(sizeof(PreviewStatus));
      std::memcpy(payload.data(), &*status, sizeof(PreviewStatus));
      return {ipc::reply(request, ipc::Kind::kAck, std::move(payload))};
    }
    case ipc::Kind::kWaitPreview: {
      if (request.payload.size() != sizeof(std::uint64_t)) {
        return {error_reply(request, "preview target size")};
      }
      std::uint64_t target = 0;
      std::memcpy(&target, request.payload.data(), sizeof(target));
      return state.runtime.wait_preview(request.camera, target)
                 ? CommandResult{ipc::reply(request, ipc::Kind::kAck)}
                 : CommandResult{error_reply(request, "preview target timeout")};
    }
    case ipc::Kind::kWaitPublish: {
      if (request.payload.size() != sizeof(std::uint64_t)) {
        return {error_reply(request, "publish target size")};
      }
      std::uint64_t target = 0;
      std::memcpy(&target, request.payload.data(), sizeof(target));
      std::unique_lock lock{state.published_mutex};
      const bool reached = state.published_condition.wait_for(
          lock, std::chrono::seconds{2},
          [&state, target] { return state.published.load() >= target; });
      return reached ? CommandResult{ipc::reply(request, ipc::Kind::kAck)}
                     : CommandResult{error_reply(request, "publish target timeout"), 4};
    }
    default:
      return {error_reply(request, "unsupported command"), 4};
  }
}
}  // namespace seeon
