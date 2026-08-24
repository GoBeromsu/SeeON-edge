#pragma once

#include "child_commands.hpp"

#include <string>
#include <vector>

namespace seeon::command_support {
[[nodiscard]] std::vector<std::uint8_t> text_payload(const std::string& value);
[[nodiscard]] ipc::Message error_reply(const ipc::Message& request, const std::string& detail);
[[nodiscard]] bool identity_matches(const ServerState& state, const ipc::Message& request);
void publish_metadata(ServerState& state, const ipc::Message& request, SourceSlot& source);
[[nodiscard]] ipc::Message status_reply(const ServerState& state, const ipc::Message& request);
}  // namespace seeon::command_support
