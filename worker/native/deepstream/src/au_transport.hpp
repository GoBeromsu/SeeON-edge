#pragma once

#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

namespace seeon {
enum class AuCodec : std::uint8_t { kH264 = 1, kH265 = 2 };
enum class AuFraming : std::uint8_t { kAnnexB = 1, kAvcc = 2 };

struct ParsedAccessUnit {
  AuCodec codec;
  AuFraming framing;
  std::int64_t pts;
  std::int64_t dts;
  std::int64_t duration;
  std::int32_t time_base_num;
  std::int32_t time_base_den;
  std::uint32_t width;
  std::uint32_t height;
  bool keyframe;
  std::string parser_caps;
  std::vector<std::uint8_t> codec_data;
  std::vector<std::uint8_t> payload;
};
class PipelineBinding {
 public:
  PipelineBinding(std::uint32_t generation, std::uint64_t epoch);
  void invalidate();
  [[nodiscard]] bool dispatch_au(
      const std::function<void(std::uint32_t, std::uint64_t, std::uint64_t)>& action);
  [[nodiscard]] bool dispatch_frame(
      const std::function<void(std::uint32_t, std::uint64_t)>& action);

 private:
  std::uint32_t generation_;
  std::uint64_t epoch_;
  std::uint64_t sequence_ = 0;
  bool live_ = true;
  std::mutex mutex_;
};
using PipelineBindingPtr = std::shared_ptr<PipelineBinding>;
using AccessUnitCallback =
    std::function<void(const std::string&, const PipelineBindingPtr&, ParsedAccessUnit)>;

struct AuEnvelope {
  std::string camera;
  std::uint32_t generation;
  std::uint64_t epoch;
  std::uint64_t sequence;
  ParsedAccessUnit unit;
};

inline constexpr std::size_t kMaxAuFrameBytes = 32U * 1024U * 1024U;

class AuSender {
 public:
  AuSender(int descriptor, std::size_t max_items, std::size_t max_bytes);
  ~AuSender();
  AuSender(const AuSender&) = delete;
  AuSender& operator=(const AuSender&) = delete;
  [[nodiscard]] bool enqueue(AuEnvelope envelope);
  void stop();
  [[nodiscard]] std::uint64_t dropped() const;

 private:
  void run();
  int descriptor_;
  std::size_t max_items_;
  std::size_t max_bytes_;
  std::size_t bytes_ = 0;
  std::uint64_t dropped_ = 0;
  std::deque<AuEnvelope> queue_;
  std::optional<AuEnvelope> gap_;
  // Per-camera reserved slot for the unit that opens a stream epoch.
  //
  // Backpressure must not destroy an epoch transition. The gap slot holds one
  // envelope and, while occupied, refuses everything - including the new
  // epoch's sequence 1, whose loss strands the receiver on the dead epoch
  // permanently (#429). These units are held per camera, keyed so a higher
  // generation always wins even though its epoch restarts at 1, and are
  // drained AHEAD of the gap as ordinary access units. Delivering them as
  // ordinary units rather than gap markers is what makes them effective:
  // NativeAuReceiver expects sequence 1 for a new (camera, generation, epoch)
  // key, so the unit passes its existing checks and the epoch is adopted with
  // no receiver-side contract change.
  //
  // Bounded by the child's own camera-identity limit; one small envelope per
  // camera, replaced rather than accumulated.
  std::map<std::string, AuEnvelope> transitions_;
  // Reservations are charged against their own budget. They are not in
  // queue_ and so are not covered by bytes_; without this a camera-sized set
  // of near-maximum sequence-1 envelopes would bypass the sender's aggregate
  // limit entirely.
  std::size_t transition_bytes_ = 0;
  bool stopped_ = false;
  mutable std::mutex mutex_;
  std::condition_variable ready_;
  std::thread thread_;
};
}  // namespace seeon
