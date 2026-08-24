#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <string>

namespace seeon {
enum class FailureScope : std::uint8_t { kSourceLocal, kFatal };
struct NativeFailure {
  std::string camera;
  std::string category;
  FailureScope scope;
};
using FrameCallback = std::function<void(const std::string&, std::uint64_t)>;
using FailureCallback = std::function<void(const NativeFailure&)>;

[[nodiscard]] bool valid_source_uri(const std::string& uri);
[[nodiscard]] NativeFailure classify_bus_failure(
    bool eos,
    unsigned int error_domain,
    int error_code,
    const std::string& element_factory,
    const std::string& camera);

class SourceRuntime {
 public:
  SourceRuntime(FrameCallback frame_callback, FailureCallback failure_callback);
  ~SourceRuntime();
  SourceRuntime(const SourceRuntime&) = delete;
  SourceRuntime& operator=(const SourceRuntime&) = delete;

  [[nodiscard]] bool add(const std::string& camera, const std::string& uri, std::string* error_code);
  [[nodiscard]] bool remove(const std::string& camera);
  [[nodiscard]] bool rebuild(const std::string& camera, std::string* error_code);
  [[nodiscard]] bool inject_eos(const std::string& camera);
  [[nodiscard]] std::size_t count() const;
  [[nodiscard]] bool custom_transform_available() const;

 private:
  class Impl;
  FrameCallback frame_callback_;
  FailureCallback failure_callback_;
  std::unique_ptr<Impl> impl_;
};
}  // namespace seeon
