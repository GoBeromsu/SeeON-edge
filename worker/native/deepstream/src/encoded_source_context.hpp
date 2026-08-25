#pragma once

#include "source_runtime.hpp"

#ifdef SEEON_HAS_GSTREAMER
#include <gst/app/gstappsink.h>
#include <gst/gst.h>

#include <atomic>
#include <condition_variable>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

namespace seeon {
struct EncodedSourceContext {
  FrameCallback frames;
  FailureCallback failures;
  AccessUnitCallback access_units;
  PreviewCallback previews;
  PipelineBindingPtr binding;
  std::string camera;
  GstElement* pipeline;
  std::uint64_t order = 0;
  bool linked = false;
  std::vector<std::uint8_t> codec_data;
  GstElement* preview_valve = nullptr;
  GstElement* decode_queue = nullptr;
  std::atomic<std::uint64_t> preview_encoded{0};
  std::atomic<std::uint64_t> preview_viewers{0};
  std::atomic<std::uint64_t> au_forwarded{0};
  std::vector<std::uint8_t> last_preview_jpeg;  // guarded by preview_mutex
  std::optional<ParsedAccessUnit> pending_duration;
  std::int64_t last_duration = 0;
  std::atomic<bool> parser_failure_latched{false};
  std::mutex sample_mutex;
  std::mutex preview_mutex;
  std::condition_variable preview_ready;
};

GstFlowReturn on_encoded_sample(GstAppSink* sink, gpointer raw);
void flush_pending_access_unit(EncodedSourceContext* context);
}  // namespace seeon
#endif
