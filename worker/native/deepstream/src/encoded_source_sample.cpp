#include "encoded_source_context.hpp"

#include "au_parser.hpp"

#include <cstdio>

#ifdef SEEON_HAS_GSTREAMER
#include <string_view>
#include <utility>

namespace seeon {
namespace {
void parser_failure(EncodedSourceContext* context, const char* reason) {
  bool expected = false;
  if (context->parser_failure_latched.compare_exchange_strong(expected, true)) {
    // Four unrelated conditions latch this one "parser" category, and only the
    // category reaches the Python supervisor. Naming the branch is what made
    // the metadata-slot and clip-selection failures diagnosable; do the same
    // here so a rebuild storm can be attributed instead of guessed at.
    std::fprintf(stderr, "seeon-parser-failure: camera=%s reason=%s\n",
                 context->camera.c_str(), reason);
    context->failures({context->camera, "parser", FailureScope::kSourceLocal});
  }
}

void emit(EncodedSourceContext* context, ParsedAccessUnit unit) {
  context->au_forwarded.fetch_add(1);
  context->access_units(context->camera, context->binding, std::move(unit));
}

std::vector<std::uint8_t> codec_data(const GstStructure* structure) {
  const GValue* value = structure == nullptr ? nullptr : gst_structure_get_value(structure, "codec_data");
  GstBuffer* buffer = value == nullptr ? nullptr : gst_value_get_buffer(value);
  GstMapInfo mapped{};
  if (buffer == nullptr || !gst_buffer_map(buffer, &mapped, GST_MAP_READ)) return {};
  std::vector<std::uint8_t> result{mapped.data, mapped.data + mapped.size};
  gst_buffer_unmap(buffer, &mapped);
  return result;
}
}  // namespace

GstFlowReturn on_encoded_sample(GstAppSink* sink, gpointer raw) {
  auto* context = static_cast<EncodedSourceContext*>(raw);
  std::lock_guard sample_lock{context->sample_mutex};
  GstSample* sample = gst_app_sink_pull_sample(sink);
  if (sample == nullptr) return GST_FLOW_ERROR;
  GstBuffer* buffer = gst_sample_get_buffer(sample);
  GstMapInfo mapped{};
  if (!GST_BUFFER_PTS_IS_VALID(buffer) || !gst_buffer_map(buffer, &mapped, GST_MAP_READ)) {
    parser_failure(context, "invalid_pts_or_map");
    gst_sample_unref(sample);
    return GST_FLOW_OK;
  }
  GstCaps* caps = gst_sample_get_caps(sample);
  gchar* caps_text = caps == nullptr ? nullptr : gst_caps_to_string(caps);
  const std::string normalized_caps = caps_text == nullptr ? std::string{} : caps_text;
  const bool h265 = normalized_caps.find("video/x-h265") != std::string::npos;
  const GstStructure* structure = caps == nullptr ? nullptr : gst_caps_get_structure(caps, 0);
  const char* stream_format =
      structure == nullptr ? nullptr : gst_structure_get_string(structure, "stream-format");
  const char* alignment =
      structure == nullptr ? nullptr : gst_structure_get_string(structure, "alignment");
  const bool byte_stream = stream_format != nullptr && std::string_view{stream_format} == "byte-stream";
  auto configuration = codec_data(structure);
  const std::size_t length_index = h265 ? 21U : 4U;
  const std::size_t length_size = configuration.size() > length_index
                                      ? (configuration[length_index] & 0x03U) + 1U
                                      : 0U;
  const auto annexb = parse_annexb(mapped.data, mapped.size, h265);
  const auto keyframe = byte_stream
                            ? annexb.keyframe
                            : parse_length_prefixed_keyframe(
                                  mapped.data, mapped.size, h265, length_size);
  if (byte_stream && configuration.empty() && !annexb.codec_data.empty()) {
    configuration = annexb.codec_data;
  }
  if (!configuration.empty()) context->codec_data = configuration;
  else configuration = context->codec_data;
  if (!keyframe.has_value() || alignment == nullptr || std::string_view{alignment} != "au") {
    parser_failure(context, "no_vcl_nal_or_alignment");
  } else {
    bool timeline_valid = true;
    gint width = 0;
    gint height = 0;
    gint fps_numerator = 0;
    gint fps_denominator = 1;
    if (structure != nullptr) {
      static_cast<void>(gst_structure_get_int(structure, "width", &width));
      static_cast<void>(gst_structure_get_int(structure, "height", &height));
      static_cast<void>(gst_structure_get_fraction(
          structure, "framerate", &fps_numerator, &fps_denominator));
    }
    const auto pts = static_cast<std::int64_t>(GST_BUFFER_PTS(buffer));
    const auto dts = GST_BUFFER_DTS_IS_VALID(buffer)
                         ? static_cast<std::int64_t>(GST_BUFFER_DTS(buffer)) : pts;
    const auto duration = GST_BUFFER_DURATION_IS_VALID(buffer)
                              ? static_cast<std::int64_t>(GST_BUFFER_DURATION(buffer))
                              : (fps_numerator > 0
                                     ? static_cast<std::int64_t>(GST_SECOND) * fps_denominator /
                                           fps_numerator : 0);
    ParsedAccessUnit unit{h265 ? AuCodec::kH265 : AuCodec::kH264,
                          byte_stream ? AuFraming::kAnnexB : AuFraming::kAvcc,
                          pts, dts, duration, 1, static_cast<std::int32_t>(GST_SECOND),
                          static_cast<std::uint32_t>(width), static_cast<std::uint32_t>(height),
                          *keyframe, normalized_caps, configuration,
                          {mapped.data, mapped.data + mapped.size}};
    if (context->pending_duration.has_value()) {
      auto pending = std::move(*context->pending_duration);
      context->pending_duration.reset();
      pending.duration = dts - pending.dts;
      if (pending.duration <= 0) {
        timeline_valid = false;
        parser_failure(context, "pending_duration_nonpositive");
      } else {
        context->last_duration = pending.duration;
        emit(context, std::move(pending));
      }
    }
    if (unit.duration > 0) {
      context->last_duration = unit.duration;
      emit(context, std::move(unit));
    } else {
      context->pending_duration = std::move(unit);
    }
    if (timeline_valid) context->parser_failure_latched = false;
  }
  g_free(caps_text);
  gst_buffer_unmap(buffer, &mapped);
  gst_sample_unref(sample);
  return GST_FLOW_OK;
}

void flush_pending_access_unit(EncodedSourceContext* context) {
  std::lock_guard sample_lock{context->sample_mutex};
  if (!context->pending_duration.has_value()) return;
  auto pending = std::move(*context->pending_duration);
  context->pending_duration.reset();
  if (context->last_duration <= 0) {
    parser_failure(context, "last_duration_nonpositive");
    return;
  }
  pending.duration = context->last_duration;
  emit(context, std::move(pending));
}
}  // namespace seeon
#endif
