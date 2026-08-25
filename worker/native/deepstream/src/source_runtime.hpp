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
using FrameCallback =
    std::function<void(const std::string&, const PipelineBindingPtr&, std::uint64_t)>;
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
