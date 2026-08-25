#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "au_transport.hpp"

namespace seeon {
enum class FailureScope : std::uint8_t { kSourceLocal, kFatal };
struct PreviewStatus {
  std::uint64_t encoded;
  std::uint64_t viewers;
};
struct NativeFailure {
  std::string camera;
  std::string category;
  FailureScope scope;
};
// One decoded RGBA frame handed to the perception stage. `rgba` is only valid
// for the duration of the callback (the GStreamer buffer stays mapped).
// Non-GStreamer builds pass a null view (all zeros) purely for lifecycle tests.
struct DecodedFrameView {
  std::uint64_t pts = 0;
  std::uint64_t source_time_ns = 0;
  int width = 0;
  int height = 0;
  int stride = 0;
  const std::uint8_t* rgba = nullptr;
};
using FrameCallback =
    std::function<void(const std::string&, const PipelineBindingPtr&, const DecodedFrameView&)>;
using FailureCallback = std::function<void(const NativeFailure&)>;
using PreviewCallback = std::function<void(
    const std::string&, std::uint64_t, std::vector<std::uint8_t>)>;

[[nodiscard]] bool valid_source_uri(const std::string& uri);
[[nodiscard]] NativeFailure classify_bus_failure(
    bool eos,
    unsigned int error_domain,
    int error_code,
    const std::string& element_factory,
    const std::string& camera);

class SourceRuntime {
 public:
  SourceRuntime(FrameCallback frame_callback, FailureCallback failure_callback,
                AccessUnitCallback access_unit_callback = {},
                PreviewCallback preview_callback = {});
  ~SourceRuntime();
  SourceRuntime(const SourceRuntime&) = delete;
  SourceRuntime& operator=(const SourceRuntime&) = delete;

  [[nodiscard]] bool add(const std::string& camera, const std::string& uri,
                         const PipelineBindingPtr& binding, std::string* error_code);
  [[nodiscard]] bool remove(const std::string& camera);
  [[nodiscard]] bool quiesce(const std::string& camera);
  [[nodiscard]] bool restart(const std::string& camera, const PipelineBindingPtr& binding,
                             std::string* error_code);
  [[nodiscard]] bool inject_eos(const std::string& camera);
  [[nodiscard]] bool set_preview_viewers(const std::string& camera, std::uint32_t viewers);
  [[nodiscard]] std::optional<PreviewStatus> preview_status(const std::string& camera) const;
  [[nodiscard]] bool wait_preview(const std::string& camera, std::uint64_t target);
  // Exact one-frame NVJPEG snapshot from the preview branch, independent of
  // viewer demand. Returns false when the camera is unknown or the bounded
  // wait for an encoded frame elapses.
  [[nodiscard]] bool snapshot_jpeg(const std::string& camera,
                                   std::vector<std::uint8_t>* jpeg);
  // Monotonic count of access units forwarded to the parent ring for this
  // camera's current pipeline (the continuous source-primary record tee).
  [[nodiscard]] std::optional<std::uint64_t> au_forwarded(const std::string& camera) const;
  [[nodiscard]] std::size_t count() const;
  [[nodiscard]] bool custom_transform_available() const;

 private:
  class Impl;
  FrameCallback frame_callback_;
  FailureCallback failure_callback_;
  AccessUnitCallback access_unit_callback_;
  PreviewCallback preview_callback_;
  std::unique_ptr<Impl> impl_;
};
}  // namespace seeon
