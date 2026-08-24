#include <gst/gst.h>

#include <chrono>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {
constexpr auto kWarmupTimeout = std::chrono::seconds{15};
constexpr char kPipeline[] =
    "videotestsrc num-buffers=1 pattern=black ! "
    "video/x-raw,width=64,height=64,format=RGBA ! nvvideoconvert ! "
    "video/x-raw(memory:NVMM),format=NV12 ! mux.sink_0 "
    "nvstreammux name=mux batch-size=1 width=64 height=64 "
    "batched-push-timeout=40000 ! fakesink sync=false";

void warmup() {
  GError* error = nullptr;
  GstElement* pipeline = gst_parse_launch(kPipeline, &error);
  if (pipeline == nullptr) {
    const std::string detail = error == nullptr ? "unknown parse failure" : error->message;
    g_clear_error(&error);
    throw std::runtime_error{"pipeline parse failed: " + detail};
  }
  if (error != nullptr) {
    const std::string detail = error->message;
    g_clear_error(&error);
    gst_object_unref(pipeline);
    throw std::runtime_error{"pipeline parse warning: " + detail};
  }
  const auto state_result = gst_element_set_state(pipeline, GST_STATE_PLAYING);
  if (state_result == GST_STATE_CHANGE_FAILURE) {
    gst_element_set_state(pipeline, GST_STATE_NULL);
    gst_object_unref(pipeline);
    throw std::runtime_error{"pipeline refused PLAYING state"};
  }

  GstBus* bus = gst_element_get_bus(pipeline);
  const auto timeout = static_cast<GstClockTime>(
      std::chrono::duration_cast<std::chrono::nanoseconds>(kWarmupTimeout).count());
  GstMessage* message = gst_bus_timed_pop_filtered(
      bus, timeout, static_cast<GstMessageType>(GST_MESSAGE_ERROR | GST_MESSAGE_EOS));
  std::string failure;
  if (message == nullptr) {
    failure = "one-source warmup timed out";
  } else if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR) {
    GError* pipeline_error = nullptr;
    gchar* debug = nullptr;
    gst_message_parse_error(message, &pipeline_error, &debug);
    failure = pipeline_error == nullptr ? "pipeline error" : pipeline_error->message;
    g_clear_error(&pipeline_error);
    g_free(debug);
  }
  if (message != nullptr) {
    gst_message_unref(message);
  }
  gst_object_unref(bus);
  gst_element_set_state(pipeline, GST_STATE_NULL);
  gst_object_unref(pipeline);
  if (!failure.empty()) {
    throw std::runtime_error{failure};
  }
}
}  // namespace

int main(int argc, char** argv) {
  gst_init(&argc, &argv);
  if (argc == 2 && std::strcmp(argv[1], "--version") == 0) {
    std::cout << "seeon-deepstream-preflight 1.0.0\n";
    return 0;
  }
  if (argc != 2 || std::strcmp(argv[1], "--warmup") != 0) {
    std::cerr << "usage: seeon-deepstream-preflight --version|--warmup\n";
    return 2;
  }
  try {
    warmup();
  } catch (const std::exception& error) {
    std::cerr << "{\"status\":\"error\",\"code\":\"warmup_failed\",\"detail\":\""
              << error.what() << "\"}\n";
    return 1;
  }
  std::cout << "{\"status\":\"ok\",\"frames\":1,\"source\":\"loopback\"}\n";
  return 0;
}
