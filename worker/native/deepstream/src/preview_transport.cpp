#include "preview_transport.hpp"

#include <sys/socket.h>
#include <unistd.h>

#include <array>
#include <cerrno>
#include <cstring>
#include <utility>

namespace seeon {
namespace {
#pragma pack(push, 1)
struct Header {
  std::array<char, 4> magic;
  std::uint32_t body_size;
  std::uint64_t sequence;
  std::uint64_t pts;
  std::uint16_t camera_size;
  std::uint32_t jpeg_size;
};
#pragma pack(pop)
static_assert(sizeof(Header) == 30);

bool send_all(int descriptor, const std::vector<std::uint8_t>& bytes) {
  std::size_t offset = 0;
  while (offset < bytes.size()) {
    const auto count = send(descriptor, bytes.data() + offset, bytes.size() - offset, MSG_NOSIGNAL);
    if (count < 0 && errno == EINTR) continue;
    if (count <= 0) return false;
    offset += static_cast<std::size_t>(count);
  }
  return true;
}

std::vector<std::uint8_t> encode(const PreviewFrame& frame) {
  Header header{{'S', 'J', 'P', '1'},
                static_cast<std::uint32_t>(frame.camera.size() + frame.jpeg.size()),
                frame.sequence,
                frame.pts,
                static_cast<std::uint16_t>(frame.camera.size()),
                static_cast<std::uint32_t>(frame.jpeg.size())};
  std::vector<std::uint8_t> bytes(sizeof(header) + header.body_size);
  std::memcpy(bytes.data(), &header, sizeof(header));
  std::memcpy(bytes.data() + sizeof(header), frame.camera.data(), frame.camera.size());
  std::memcpy(bytes.data() + sizeof(header) + frame.camera.size(), frame.jpeg.data(),
              frame.jpeg.size());
  return bytes;
}
}  // namespace

PreviewSender::PreviewSender(int descriptor)
    : descriptor_(descriptor), thread_(&PreviewSender::run, this) {}
PreviewSender::~PreviewSender() { stop(); }

bool PreviewSender::publish(std::string camera, std::uint64_t pts,
                            std::vector<std::uint8_t> jpeg) {
  if (camera.empty() || camera.size() > UINT16_MAX || jpeg.empty() ||
      camera.size() + jpeg.size() > kMaxPreviewBytes) return false;
  std::lock_guard lock{mutex_};
  if (stopped_) return false;
  latest_ = PreviewFrame{std::move(camera), ++sequence_, pts, std::move(jpeg)};
  ready_.notify_one();
  return true;
}

void PreviewSender::run() {
  while (true) {
    PreviewFrame frame{};
    {
      std::unique_lock lock{mutex_};
      ready_.wait(lock, [this] { return stopped_ || latest_.has_value(); });
      if (stopped_ && !latest_.has_value()) return;
      frame = std::move(*latest_);
      latest_.reset();
    }
    if (!send_all(descriptor_, encode(frame))) {
      std::lock_guard lock{mutex_};
      stopped_ = true;
      latest_.reset();
      return;
    }
  }
}

void PreviewSender::stop() {
  {
    std::lock_guard lock{mutex_};
    stopped_ = true;
    latest_.reset();
  }
  shutdown(descriptor_, SHUT_WR);
  ready_.notify_all();
  if (thread_.joinable()) thread_.join();
}
}  // namespace seeon
