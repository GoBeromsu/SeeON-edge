#include "source_runtime.hpp"

#include <cstdlib>
#include <iostream>
#include <memory>
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
  check(seeon::valid_source_uri("loopback://camera-a"), "loopback URI rejected");
  const std::string credentialed_uri =
      "rtsp://user:p'ass" + std::string(1, '@') + "camera.example/live";
  check(seeon::valid_source_uri(credentialed_uri), "quote-containing RTSP data rejected");
  check(!seeon::valid_source_uri("rtsp:///missing-host"), "hostless RTSP URI accepted");
  check(!seeon::valid_source_uri("rtsp://?missing-host"), "query-only RTSP URI accepted");
  check(!seeon::valid_source_uri("file:///tmp/input"), "file URI accepted");
  check(!seeon::valid_source_uri("http://camera.example/live"), "HTTP URI accepted");
  check(!seeon::valid_source_uri("rtsp://camera.example/live\nfilesink"),
        "control character accepted");
  check(!seeon::valid_source_uri("rtsp://camera.example/" + std::string(4096, 'x')),
        "oversized URI accepted");

  std::vector<std::string> frames;
  std::vector<seeon::NativeFailure> failures;
  seeon::SourceRuntime runtime(
      [&frames](const std::string& camera, const seeon::PipelineBindingPtr&, std::uint64_t) {
        frames.push_back(camera);
      },
      [&failures](const seeon::NativeFailure& failure) { failures.push_back(failure); });
  std::string error_code;
  const auto binding = std::make_shared<seeon::PipelineBinding>(1, 1);
  check(!runtime.add("camera-file", "file:///tmp/input", binding, &error_code),
        "invalid source was added");
  check(error_code == "source_uri_invalid", "invalid source error taxonomy changed");
  check(runtime.add("camera-a", "loopback://camera-a", binding, &error_code),
        "source add failed");
  check(runtime.count() == 1, "source count mismatch");
  check(!frames.empty() && frames.back() == "camera-a", "source frame callback absent");
  check(runtime.inject_eos("camera-a"), "EOS injection failed");
  check(failures.size() == 1, "EOS failure callback absent");
  check(failures.front().camera == "camera-a", "EOS camera identity absent");
  check(failures.front().category == "eos", "EOS category mismatch");
  check(failures.front().scope == seeon::FailureScope::kSourceLocal, "EOS scope mismatch");
  check(runtime.remove("camera-a"), "source removal failed");
  return 0;
}
