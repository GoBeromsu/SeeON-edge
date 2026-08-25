#include "source_runtime.hpp"

#include <gst/gst.h>

#include <cstdlib>
#include <iostream>
#include <string>

namespace {
void check(bool condition, const std::string& message) {
  if (!condition) {
    std::cerr << message << '\n';
    std::abort();
  }
}
}  // namespace

int main() {
  gst_init(nullptr, nullptr);
  const seeon::NativeFailure eos =
      seeon::classify_bus_failure(true, 0, 0, "uridecodebin", "camera-a");
  check(eos.category == "eos" && eos.scope == seeon::FailureScope::kSourceLocal,
        "real-bus EOS classification changed");

  const seeon::NativeFailure camera_message = seeon::classify_bus_failure(
      false, GST_STREAM_ERROR, GST_STREAM_ERROR_FAILED, "uridecodebin", "cuda.context.camera");
  check(camera_message.scope == seeon::FailureScope::kSourceLocal,
        "camera text incorrectly changed fatal scope");

  const seeon::NativeFailure shared = seeon::classify_bus_failure(
      false, GST_LIBRARY_ERROR, GST_LIBRARY_ERROR_FAILED, "seeonperceptiontransform", "camera-a");
  check(shared.scope == seeon::FailureScope::kFatal && shared.category == "shared_pipeline",
        "shared GStreamer failure was not fatal");
  return 0;
}
