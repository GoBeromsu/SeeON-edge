#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace seeon {
inline constexpr std::size_t kMaxPreviewBytes = 2U * 1024U * 1024U;
struct PreviewFrame {
  std::string camera;
  std::uint64_t sequence;
  std::uint64_t pts;
  std::vector<std::uint8_t> jpeg;
};

class PreviewSender {
 public:
  explicit PreviewSender(int descriptor);
  ~PreviewSender();
  PreviewSender(const PreviewSender&) = delete;
  PreviewSender& operator=(const PreviewSender&) = delete;
  [[nodiscard]] bool publish(std::string camera, std::uint64_t pts,
                             std::vector<std::uint8_t> jpeg);
  void stop();

 private:
  void run();
  int descriptor_;
  std::uint64_t sequence_ = 0;
  std::optional<PreviewFrame> latest_;
  bool stopped_ = false;
  std::mutex mutex_;
  std::condition_variable ready_;
  std::thread thread_;
};
}  // namespace seeon
