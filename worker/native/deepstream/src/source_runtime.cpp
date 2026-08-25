#include "source_runtime.hpp"

#include "encoded_source_branch.hpp"
#include "source_bus.hpp"

#include <algorithm>
#include <cctype>
#include <map>
#include <utility>

#ifdef SEEON_HAS_GSTREAMER
#include <gst/app/gstappsink.h>
#include <gst/base/gstbasetransform.h>
#include <gst/gst.h>
#include <gst/video/video.h>

namespace {
typedef struct _SeeonPerceptionTransform {
  GstBaseTransform parent;
} SeeonPerceptionTransform;
typedef struct _SeeonPerceptionTransformClass {
  GstBaseTransformClass parent_class;
} SeeonPerceptionTransformClass;
G_DEFINE_TYPE(SeeonPerceptionTransform, seeon_perception_transform, GST_TYPE_BASE_TRANSFORM)

GstFlowReturn transform_in_place(GstBaseTransform*, GstBuffer*) { return GST_FLOW_OK; }
void seeon_perception_transform_class_init(SeeonPerceptionTransformClass* klass) {
  auto* element = GST_ELEMENT_CLASS(klass);
  auto* transform = GST_BASE_TRANSFORM_CLASS(klass);
  gst_element_class_set_static_metadata(element, "SeeON perception transform", "Filter/Metadata",
                                        "Attaches compact SeeON perception metadata", "SeniorAILab");
  GstCaps* caps = gst_caps_new_any();
  gst_element_class_add_pad_template(
      element, gst_pad_template_new("sink", GST_PAD_SINK, GST_PAD_ALWAYS, caps));
  gst_element_class_add_pad_template(
      element, gst_pad_template_new("src", GST_PAD_SRC, GST_PAD_ALWAYS, caps));
  gst_caps_unref(caps);
  transform->transform_ip = transform_in_place;
  transform->passthrough_on_same_caps = true;
}
void seeon_perception_transform_init(SeeonPerceptionTransform* transform) {
  gst_base_transform_set_in_place(GST_BASE_TRANSFORM(transform), true);
}

struct SourceContext {
  seeon::FrameCallback frame_callback;
  seeon::PipelineBindingPtr binding;
  std::string camera;
};

GstFlowReturn on_generic_sample(GstAppSink* sink, gpointer raw) {
  auto* context = static_cast<SourceContext*>(raw);
  GstSample* sample = gst_app_sink_pull_sample(sink);
  if (sample == nullptr) return GST_FLOW_ERROR;
  GstCaps* caps = gst_sample_get_caps(sample);
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  GstVideoInfo info;
  gst_video_info_init(&info);
  if (caps == nullptr || buffer == nullptr || !gst_video_info_from_caps(&info, caps)) {
    gst_sample_unref(sample);
    return GST_FLOW_OK;
  }
  GstVideoFrame frame;
  if (!gst_video_frame_map(&frame, &info, buffer, GST_MAP_READ)) {
    gst_sample_unref(sample);
    return GST_FLOW_OK;
  }
  const seeon::DecodedFrameView view{
      GST_BUFFER_PTS_IS_VALID(buffer) ? GST_BUFFER_PTS(buffer) : 0,
      static_cast<std::uint64_t>(g_get_real_time()) * 1000ULL,
      GST_VIDEO_INFO_WIDTH(&info),
      GST_VIDEO_INFO_HEIGHT(&info),
      GST_VIDEO_FRAME_PLANE_STRIDE(&frame, 0),
      static_cast<const std::uint8_t*>(GST_VIDEO_FRAME_PLANE_DATA(&frame, 0)),
  };
  context->frame_callback(context->camera, context->binding, view);
  gst_video_frame_unmap(&frame);
  gst_sample_unref(sample);
  return GST_FLOW_OK;
}
void destroy_context(gpointer raw) { delete static_cast<SourceContext*>(raw); }

void on_decode_pad(GstElement*, GstPad* output, gpointer raw_convert) {
  GstPad* input = gst_element_get_static_pad(GST_ELEMENT(raw_convert), "sink");
  if (!gst_pad_is_linked(input)) {
    static_cast<void>(gst_pad_link(output, input));
  }
  gst_object_unref(input);
}

}  // namespace
#endif

namespace seeon {
class SourceRuntime::Impl {
 public:
#ifdef SEEON_HAS_GSTREAMER
  struct Source {
    std::string uri;
    GstElement* pipeline;
    std::uint32_t preview_viewers;
    PipelineBindingPtr binding;
  };
  std::map<std::string, Source> sources;
  std::map<std::string, std::uint32_t> preview_demands;
  bool transform_available = false;
#else
  struct Source { std::string uri; PipelineBindingPtr binding; };
  std::map<std::string, Source> sources;
  bool transform_available = true;
#endif
};

SourceRuntime::SourceRuntime(FrameCallback frame_callback, FailureCallback failure_callback,
                             AccessUnitCallback access_unit_callback,
                             PreviewCallback preview_callback)
    : frame_callback_(std::move(frame_callback)),
      failure_callback_(std::move(failure_callback)),
      access_unit_callback_(std::move(access_unit_callback)),
      preview_callback_(std::move(preview_callback)),
      impl_(std::make_unique<Impl>()) {
#ifdef SEEON_HAS_GSTREAMER
  gst_init(nullptr, nullptr);
  impl_->transform_available = gst_element_register(
      nullptr, "seeonperceptiontransform", GST_RANK_NONE, seeon_perception_transform_get_type());
#endif
}

SourceRuntime::~SourceRuntime() {
#ifdef SEEON_HAS_GSTREAMER
  for (auto& [camera, source] : impl_->sources) {
    static_cast<void>(camera);
    source.binding->invalidate();
    if (source.pipeline != nullptr) {
      quiesce_encoded_pipeline(source.pipeline);
      gst_element_set_state(source.pipeline, GST_STATE_NULL);
      gst_object_unref(source.pipeline);
    }
  }
#endif
}

#ifdef SEEON_HAS_GSTREAMER
GstElement* build_pipeline(const std::string& camera, const std::string& uri,
                           const FrameCallback& frames, const FailureCallback& failures,
                           const AccessUnitCallback& access_units,
                           const PreviewCallback& previews,
                           const PipelineBindingPtr& binding, std::string* error_code) {
  if (uri.starts_with("rtsp://")) {
    return build_encoded_rtsp_pipeline(
        camera, uri, frames, failures, access_units, previews, binding, error_code);
  }
  GstElement* pipeline = gst_pipeline_new(nullptr);
  GstElement* source = gst_element_factory_make(uri.starts_with("loopback://")
                                                    ? "videotestsrc" : "uridecodebin", nullptr);
  GstElement* convert = gst_element_factory_make("videoconvert", nullptr);
  GstElement* rgba = gst_element_factory_make("capsfilter", nullptr);
  GstElement* transform = gst_element_factory_make("seeonperceptiontransform", nullptr);
  GstElement* sink = gst_element_factory_make("appsink", nullptr);
  if (pipeline == nullptr || source == nullptr || convert == nullptr || rgba == nullptr ||
      transform == nullptr || sink == nullptr) {
    *error_code = "camera_id=" + camera + " element_unavailable";
    GstElement* elements[] = {source, convert, rgba, transform, sink};
    for (GstElement* element : elements) {
      if (element != nullptr) {
        static_cast<void>(gst_object_ref_sink(element));
        gst_object_unref(element);
      }
    }
    if (pipeline != nullptr) gst_object_unref(pipeline);
    return nullptr;
  }
  if (uri.starts_with("loopback://")) {
    g_object_set(source, "is-live", TRUE, "pattern", 2, nullptr);
  } else {
    g_object_set(source, "uri", uri.c_str(), nullptr);
    g_signal_connect(source, "pad-added", G_CALLBACK(on_decode_pad), convert);
  }
  GstCaps* rgba_caps = gst_caps_from_string("video/x-raw,format=RGBA");
  g_object_set(rgba, "caps", rgba_caps, nullptr);
  g_object_set(sink, "emit-signals", TRUE, "sync", FALSE, "max-buffers", 1U, "drop", TRUE,
               nullptr);
  gst_caps_unref(rgba_caps);
  gst_bin_add_many(GST_BIN(pipeline), source, convert, rgba, transform, sink, nullptr);
  const bool downstream = gst_element_link_many(convert, rgba, transform, sink, nullptr);
  const bool upstream = !uri.starts_with("loopback://") || gst_element_link(source, convert);
  if (!downstream || !upstream) {
    *error_code = "element_link_failed";
    gst_object_unref(pipeline);
    return nullptr;
  }
  auto* context = new SourceContext{frames, binding, camera};
  g_signal_connect_data(sink, "new-sample", G_CALLBACK(on_generic_sample), context,
                        [](gpointer raw, GClosure*) { destroy_context(raw); },
                        static_cast<GConnectFlags>(0));
  attach_source_bus_handler(pipeline, camera, failures);
  if (gst_element_set_state(pipeline, GST_STATE_PLAYING) == GST_STATE_CHANGE_FAILURE) {
    *error_code = "pipeline_start_failed";
    gst_object_unref(pipeline);
    return nullptr;
  }
  return pipeline;
}
#endif

bool SourceRuntime::add(const std::string& camera, const std::string& uri,
                        const PipelineBindingPtr& binding, std::string* error_code) {
  if (!valid_source_uri(uri)) { *error_code = "source_uri_invalid"; return false; }
  if (impl_->sources.contains(camera)) { *error_code = "source_exists"; return false; }
#ifdef SEEON_HAS_GSTREAMER
  GstElement* pipeline = build_pipeline(
      camera, uri, frame_callback_, failure_callback_, access_unit_callback_, preview_callback_,
      binding, error_code);
  if (pipeline == nullptr) return false;
  const auto viewers = impl_->preview_demands[camera];
  static_cast<void>(set_encoded_preview_viewers(pipeline, viewers));
  impl_->sources.emplace(camera, Impl::Source{uri, pipeline, viewers, binding});
#else
  impl_->sources.emplace(camera, Impl::Source{uri, binding});
  frame_callback_(camera, binding, DecodedFrameView{});
#endif
  return true;
}

bool SourceRuntime::remove(const std::string& camera) {
  const auto found = impl_->sources.find(camera);
  if (found == impl_->sources.end()) return false;
  if (!quiesce(camera)) return false;
  impl_->sources.erase(found);
  return true;
}

bool SourceRuntime::quiesce(const std::string& camera) {
  const auto found = impl_->sources.find(camera);
  if (found == impl_->sources.end()) return false;
  found->second.binding->invalidate();
#ifdef SEEON_HAS_GSTREAMER
  if (found->second.pipeline != nullptr) {
    quiesce_encoded_pipeline(found->second.pipeline);
    static_cast<void>(gst_element_set_state(found->second.pipeline, GST_STATE_NULL));
    static_cast<void>(gst_element_get_state(
        found->second.pipeline, nullptr, nullptr, 2 * GST_SECOND));
    gst_object_unref(found->second.pipeline);
    found->second.pipeline = nullptr;
  }
#endif
  return true;
}

bool SourceRuntime::restart(const std::string& camera, const PipelineBindingPtr& binding,
                            std::string* error_code) {
  const auto found = impl_->sources.find(camera);
  if (found == impl_->sources.end()) { *error_code = "source_unknown"; return false; }
#ifdef SEEON_HAS_GSTREAMER
  GstElement* replacement = build_pipeline(
      camera, found->second.uri, frame_callback_, failure_callback_, access_unit_callback_,
      preview_callback_, binding, error_code);
  if (replacement == nullptr) return false;
  static_cast<void>(set_encoded_preview_viewers(replacement, found->second.preview_viewers));
  found->second.pipeline = replacement;
  found->second.binding = binding;
#else
  found->second.binding = binding;
  frame_callback_(camera, binding, DecodedFrameView{});
#endif
  return true;
}

bool SourceRuntime::inject_eos(const std::string& camera) {
#ifdef SEEON_HAS_GSTREAMER
  const auto found = impl_->sources.find(camera);
  return found != impl_->sources.end() && found->second.pipeline != nullptr &&
         gst_element_send_event(found->second.pipeline, gst_event_new_eos());
#else
  failure_callback_({camera, "eos", FailureScope::kSourceLocal});
  return impl_->sources.contains(camera);
#endif
}
bool SourceRuntime::set_preview_viewers(const std::string& camera, std::uint32_t viewers) {
#ifdef SEEON_HAS_GSTREAMER
  impl_->preview_demands[camera] = viewers;
  const auto found = impl_->sources.find(camera);
  if (found == impl_->sources.end()) return true;
  if (found->second.pipeline != nullptr &&
      !set_encoded_preview_viewers(found->second.pipeline, viewers)) return false;
  found->second.preview_viewers = viewers;
  return true;
#else
  static_cast<void>(viewers);
  return impl_->sources.contains(camera);
#endif
}

bool SourceRuntime::wait_preview(const std::string& camera, std::uint64_t target) {
#ifdef SEEON_HAS_GSTREAMER
  const auto found = impl_->sources.find(camera);
  return found != impl_->sources.end() && found->second.pipeline != nullptr &&
         wait_encoded_preview(found->second.pipeline, target);
#else
  static_cast<void>(camera);
  static_cast<void>(target);
  return false;
#endif
}

std::optional<PreviewStatus> SourceRuntime::preview_status(const std::string& camera) const {
#ifdef SEEON_HAS_GSTREAMER
  const auto found = impl_->sources.find(camera);
  return found == impl_->sources.end() || found->second.pipeline == nullptr
             ? std::nullopt
             : encoded_preview_status(found->second.pipeline);
#else
  return impl_->sources.contains(camera) ? std::optional{PreviewStatus{0, 0}} : std::nullopt;
#endif
}

bool SourceRuntime::snapshot_jpeg(const std::string& camera,
                                  std::vector<std::uint8_t>* jpeg) {
#ifdef SEEON_HAS_GSTREAMER
  const auto found = impl_->sources.find(camera);
  return found != impl_->sources.end() && found->second.pipeline != nullptr &&
         snapshot_encoded_preview(found->second.pipeline, jpeg);
#else
  if (!impl_->sources.contains(camera)) return false;
  jpeg->assign({0xFF, 0xD8, 0xFF, 0xD9});  // minimal JPEG for lifecycle tests
  return true;
#endif
}

std::optional<std::uint64_t> SourceRuntime::au_forwarded(const std::string& camera) const {
#ifdef SEEON_HAS_GSTREAMER
  const auto found = impl_->sources.find(camera);
  return found == impl_->sources.end() || found->second.pipeline == nullptr
             ? std::nullopt
             : std::optional{encoded_au_forwarded(found->second.pipeline)};
#else
  return impl_->sources.contains(camera) ? std::optional<std::uint64_t>{0} : std::nullopt;
#endif
}

std::size_t SourceRuntime::count() const { return impl_->sources.size(); }
bool SourceRuntime::custom_transform_available() const { return impl_->transform_available; }
}  // namespace seeon
