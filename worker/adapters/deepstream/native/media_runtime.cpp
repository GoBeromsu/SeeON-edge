#include "media_internal.h"
#include <cerrno>
#include <climits>
#include <initializer_list>
#include <utility>

using namespace seeon_media;
namespace {
CallbackData *callback_data(SeeonMedia &m, uint32_t source, CallbackData *handle = nullptr) {
  auto registration = std::make_shared<CallbackRegistration>();
  registration->lifetime = m.lifetime;
  registration->owner = &m;
  registration->source = source;
  auto *data = new CallbackData(registration);
  m.lifetime->registrations.fetch_add(1);
  if (handle) *handle = std::move(registration);
  return data;
}
void callback_destroy(gpointer data) noexcept {
  auto *box = static_cast<CallbackData *>(data);
  auto registration = std::move(*box);
  delete box;
  registration->entries.retire();
  registration->detached.store(true);
  // No owner access here. This acknowledgement comes from the SDK's final
  // invocation reference, not from observing zero callbacks at an instant.
  registration->lifetime->registrations.fetch_sub(1);
}
void closure_destroy(gpointer data, GClosure *) noexcept { callback_destroy(data); }

std::string copied(const char *text, size_t limit = SEEON_MEDIA_PATH_BYTES) {
  require(text != nullptr, SEEON_MEDIA_ERROR_CONFIG);
  const size_t length = strnlen(text, limit);
  require(length > 0 && length < limit, SEEON_MEDIA_ERROR_CONFIG);
  return std::string(text, length);
}
std::string key_text(GKeyFile *file, const char *key) {
  std::unique_ptr<gchar, decltype(&g_free)> value(
      g_key_file_get_string(file, "property", key, nullptr), &g_free);
  return value ? std::string(value.get()) : std::string();
}
int key_integer(GKeyFile *file, const char *key) {
  GError *error = nullptr;
  int value = g_key_file_get_integer(file, "property", key, &error);
  const bool valid = error == nullptr;
  if (error) g_error_free(error);
  require(valid, SEEON_MEDIA_ERROR_CONFIG);
  return value;
}
void admit_inference(const SeeonMedia &m) {
  std::unique_ptr<GKeyFile, decltype(&g_key_file_unref)> file(g_key_file_new(), &g_key_file_unref);
  require(g_key_file_load_from_file(file.get(), m.infer_path.c_str(), G_KEY_FILE_NONE, nullptr),
          SEEON_MEDIA_ERROR_CONFIG);
  for (const char *key : {"onnx-file", "model-file", "proto-file", "uff-file",
                         "tlt-encoded-model", "custom-network-config", "engine-create-func-name"})
    require(!g_key_file_has_key(file.get(), "property", key, nullptr), SEEON_MEDIA_ERROR_CONFIG);
  const auto engine = key_text(file.get(), "model-engine-file");
  const auto parser = key_text(file.get(), "custom-lib-path");
  require(!engine.empty() && g_file_test(engine.c_str(), G_FILE_TEST_IS_REGULAR) &&
          !parser.empty() && g_file_test(parser.c_str(), G_FILE_TEST_IS_REGULAR) &&
          key_text(file.get(), "parse-bbox-func-name") == "NvDsInferParseCustomYolo26Pose" &&
          key_text(file.get(), "output-blob-names") == "output0" &&
          key_integer(file.get(), "gie-unique-id") == 1 &&
          key_integer(file.get(), "output-tensor-meta") == 1 &&
          key_integer(file.get(), "gpu-id") == 0 &&
          key_integer(file.get(), "batch-size") == int(m.config.source_count),
          SEEON_MEDIA_ERROR_CONFIG);
}
void copy_config(SeeonMedia &m, const SeeonMediaConfig &c) {
  require(c.abi_version == SEEON_MEDIA_ABI_VERSION && c.struct_size == sizeof(c) &&
          c.sources && c.source_count > 0 && c.source_count <= SEEON_MEDIA_MAX_SOURCES &&
          c.mux_width > 0 && c.mux_width <= INT_MAX && c.mux_height > 0 && c.mux_height <= INT_MAX &&
          c.mux_batch_timeout_us > 0 && c.mux_batch_timeout_us <= INT_MAX && c.mux_live_source <= 1 &&
          c.tracker_width > 0 && c.tracker_width <= INT_MAX &&
          c.tracker_height > 0 && c.tracker_height <= INT_MAX &&
          c.record_cache_seconds > 0 && c.record_capacity > 0 &&
          c.record_capacity <= SEEON_MEDIA_MAX_RECORDS && c.queue_max_buffers > 0 &&
          c.queue_max_buffers <= 64 && c.preview_enabled <= 1 && c.allow_file_uris <= 1 &&
          (c.preview_enabled ? c.max_preview_bytes >= 4 &&
             c.max_preview_bytes <= SEEON_MEDIA_MAX_PREVIEW_BYTES : c.max_preview_bytes == 0),
          SEEON_MEDIA_ERROR_CONFIG);
  m.config = c;
  m.config.sources = nullptr;
  m.config.infer_config_path = m.config.tracker_config_path = nullptr;
  m.config.tracker_library_path = m.config.record_directory = nullptr;
  m.infer_path = copied(c.infer_config_path);
  m.tracker_config = copied(c.tracker_config_path);
  m.tracker_library = copied(c.tracker_library_path);
  m.record_directory = copied(c.record_directory);
  require(g_file_test(m.tracker_config.c_str(), G_FILE_TEST_IS_REGULAR) &&
          g_file_test(m.tracker_library.c_str(), G_FILE_TEST_IS_REGULAR), SEEON_MEDIA_ERROR_CONFIG);
  for (uint32_t i = 0; i < c.source_count; ++i) {
    const auto &input = c.sources[i];
    auto &source = m.sources[i];
    require(input.source_id == i && input.binding.token != 0, SEEON_MEDIA_ERROR_CONFIG);
    source.index = i; source.binding = input.binding;
    source.uri = copied(input.uri);
    source.rtsp = source.uri.rfind("rtsp://", 0) == 0 || source.uri.rfind("rtsps://", 0) == 0;
    require(source.rtsp || (c.allow_file_uris && source.uri.rfind("file://", 0) == 0),
            SEEON_MEDIA_ERROR_CONFIG);
    if (source.rtsp) {
      require(g_file_test(m.record_directory.c_str(), G_FILE_TEST_IS_DIR), SEEON_MEDIA_ERROR_CONFIG);
      source.prefix = copied(input.record_prefix, 128);
      require(source.prefix != "." && source.prefix != ".." &&
              source.prefix.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789._-") ==
                  std::string::npos, SEEON_MEDIA_ERROR_CONFIG);
    }
    for (uint32_t j = 0; j < i; ++j)
      require(source.binding.token != m.sources[j].binding.token &&
              (!source.rtsp || !m.sources[j].rtsp || source.prefix != m.sources[j].prefix),
              SEEON_MEDIA_ERROR_CONFIG);
  }
  m.records.reset(new RecordSlot[c.record_capacity]);
  if (c.preview_enabled) m.preview.bytes.resize(c.max_preview_bytes);
  admit_inference(m);
}
GstElement *element(GstElement *parent, const char *factory, const char *name) {
  auto *result = gst_element_factory_make(factory, name);
  require(result != nullptr, SEEON_MEDIA_ERROR_PLUGIN);
  if (!gst_bin_add(GST_BIN(parent), result)) {
    gst_object_unref(result);
    throw Failure{SEEON_MEDIA_ERROR_PLUGIN};
  }
  return result;
}
void properties(GstElement *element, std::initializer_list<const char *> names) {
  for (const char *name : names) {
    auto *spec = g_object_class_find_property(G_OBJECT_GET_CLASS(element), name);
    require(spec && (spec->flags & G_PARAM_WRITABLE), SEEON_MEDIA_ERROR_PLUGIN);
  }
}
void exact_numbers(GstElement *element, std::initializer_list<std::pair<const char *, uint64_t>> values) {
  for (const auto &entry : values) {
    auto *spec = g_object_class_find_property(G_OBJECT_GET_CLASS(element), entry.first);
    require(spec && (spec->flags & G_PARAM_READABLE), SEEON_MEDIA_ERROR_PLUGIN);
    GValue value = G_VALUE_INIT;
    g_value_init(&value, G_PARAM_SPEC_VALUE_TYPE(spec));
    g_object_get_property(G_OBJECT(element), entry.first, &value);
    uint64_t actual = UINT64_MAX;
    if (G_VALUE_HOLDS_UINT(&value)) actual = g_value_get_uint(&value);
    else if (G_VALUE_HOLDS_INT(&value)) actual = uint64_t(g_value_get_int(&value));
    else if (G_VALUE_HOLDS_UINT64(&value)) actual = g_value_get_uint64(&value);
    else if (G_VALUE_HOLDS_INT64(&value)) actual = uint64_t(g_value_get_int64(&value));
    else if (G_VALUE_HOLDS_BOOLEAN(&value)) actual = g_value_get_boolean(&value) != FALSE;
    else if (G_VALUE_HOLDS_ENUM(&value)) actual = uint64_t(g_value_get_enum(&value));
    else if (G_VALUE_HOLDS_FLAGS(&value)) actual = g_value_get_flags(&value);
    g_value_unset(&value);
    require(actual == entry.second, SEEON_MEDIA_ERROR_CONFIG);
  }
}
void queue_limit(SeeonMedia &m, GstElement *queue, bool leaky) {
  g_object_set(queue, "max-size-buffers", m.config.queue_max_buffers, "max-size-bytes", 0u,
               "max-size-time", guint64(0), "leaky", leaky ? 2 : 0, nullptr);
  exact_numbers(queue, {{"max-size-buffers", m.config.queue_max_buffers}, {"max-size-bytes", 0},
                       {"max-size-time", 0}, {"leaky", leaky ? 2u : 0u}});
}
void tee_link(SeeonMedia &m, GstElement *target, uint32_t index) {
  m.tee_pads[index] = gst_element_request_pad_simple(m.tee, "src_%u");
  auto *sink = gst_element_get_static_pad(target, "sink");
  const bool linked = m.tee_pads[index] && sink &&
      gst_pad_link(m.tee_pads[index], sink) == GST_PAD_LINK_OK;
  if (sink) gst_object_unref(sink);
  require(linked, SEEON_MEDIA_ERROR_LINK);
}
void signal_contract(GstElement *source, const char *name, std::initializer_list<GType> types) {
  GSignalQuery query{};
  g_signal_query(g_signal_lookup(name, G_OBJECT_TYPE(source)), &query);
  require(query.signal_id && query.return_type == G_TYPE_NONE && query.n_params == types.size(),
          SEEON_MEDIA_ERROR_PLUGIN);
  uint32_t index = 0;
  for (auto type : types)
    require((query.param_types[index++] & ~G_SIGNAL_TYPE_STATIC_SCOPE) == type,
            SEEON_MEDIA_ERROR_PLUGIN);
}
bool video_caps(const GstCaps *caps) noexcept {
  return caps && gst_caps_is_fixed(caps) && gst_caps_get_size(caps) == 1 &&
      gst_structure_has_name(gst_caps_get_structure(caps, 0), "video/x-raw") &&
      !gst_caps_features_is_any(gst_caps_get_features(caps, 0)) &&
      gst_caps_features_contains(gst_caps_get_features(caps, 0), "memory:NVMM");
}
GstPadProbeReturn source_caps(GstPad *, GstPadProbeInfo *info, gpointer data) noexcept {
  CallbackGuard guard(data);
  if (!guard) return GST_PAD_PROBE_DROP;
  auto &m = guard.media();
  auto &source = m.sources[guard.source_index()];
  if (m.stop_requested.load() || m.fatal.load()) return GST_PAD_PROBE_DROP;
  if (GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM) {
    auto *event = GST_PAD_PROBE_INFO_EVENT(info);
    if (GST_EVENT_TYPE(event) == GST_EVENT_CAPS) {
      GstCaps *caps = nullptr;
      gst_event_parse_caps(event, &caps);
      source.negotiated.store(video_caps(caps));
      if (!source.negotiated.load()) {
        m.fail(SEEON_MEDIA_ERROR_SOURCE_CAPS);
        return GST_PAD_PROBE_DROP;
      }
    }
  }
  if ((GST_PAD_PROBE_INFO_TYPE(info) & GST_PAD_PROBE_TYPE_BUFFER) &&
      !source.negotiated.load()) {
    m.fail(SEEON_MEDIA_ERROR_SOURCE_CAPS);
    return GST_PAD_PROBE_DROP;
  }
  return GST_PAD_PROBE_OK;
}
void pad_added(GstElement *, GstPad *pad, gpointer data) noexcept {
  CallbackGuard guard(data);
  if (!guard) return;
  auto &m = guard.media();
  auto &source = m.sources[guard.source_index()];
  try {
    if (m.stop_requested.load() || m.fatal.load()) return;
    // nvurisrcbin may expose its ghost pad before fixed caps exist. Linking
    // enables negotiation; the probe admits no buffer until real NVMM video
    // CAPS have arrived. A pad name or query-capability is not GPU evidence.
    // Own the reference before publication. Only the winner may install a
    // producer; disconnect waits for this lease before reading its probe ID.
    auto *owned = GST_PAD(gst_object_ref(pad));
    GstPad *empty_pad = nullptr;
    if (!source.video_pad.compare_exchange_strong(empty_pad, owned)) {
      gst_object_unref(owned);
      m.fail(SEEON_MEDIA_ERROR_LINK);
      return;
    }
    auto *caps = gst_pad_get_current_caps(pad);
    source.negotiated.store(video_caps(caps));
    if (caps) gst_caps_unref(caps);
    source.video_probe = add_probe(m,
        pad, GstPadProbeType(GST_PAD_PROBE_TYPE_EVENT_DOWNSTREAM | GST_PAD_PROBE_TYPE_BUFFER),
        source_caps, source.index);
    if (!source.video_probe) { m.fail(SEEON_MEDIA_ERROR_LINK); return; }
    bool empty = false;
    if (!source.linked.compare_exchange_strong(empty, true) ||
        gst_pad_link(pad, source.mux_pad) != GST_PAD_LINK_OK)
      m.fail(SEEON_MEDIA_ERROR_LINK);
  } catch (...) { m.fail(SEEON_MEDIA_ERROR_EXCEPTION); }
}
void pad_removed(GstElement *, GstPad *pad, gpointer data) noexcept {
  CallbackGuard guard(data);
  if (!guard) return;
  auto &m = guard.media();
  auto &source = m.sources[guard.source_index()];
  try {
    // Removal can overlap pad-added before its owned pointer is published.
    // Fail closed even then; pointer identity only governs per-pad flags.
    if (!m.stop_requested.load()) m.fail(SEEON_MEDIA_ERROR_LINK);
    if (pad != source.video_pad.load()) return;
    source.linked.store(false);
    source.negotiated.store(false);
  } catch (...) { m.fail(SEEON_MEDIA_ERROR_EXCEPTION); }
}
void build_source(SeeonMedia &m, Source &source) {
  const auto name = "source-" + std::to_string(source.index);
  auto *s = source.element = element(m.pipeline, "nvurisrcbin", name.c_str());
  properties(s, {"uri", "type", "source-id", "disable-audio", "cudadec-memtype", "gpu-id"});
  g_object_set(s, "uri", source.uri.c_str(), "type", source.rtsp ? 2 : 1,
      "source-id", int(source.index), "disable-audio", TRUE, "cudadec-memtype", 0, "gpu-id", 0u, nullptr);
  if (source.rtsp) {
    properties(s, {"select-rtp-protocol", "latency", "init-rtsp-reconnect-interval",
        "rtsp-reconnect-interval", "smart-record", "smart-rec-cache", "smart-rec-container",
        "smart-rec-mode", "smart-rec-dir-path", "smart-rec-file-prefix", "smart-rec-default-duration"});
    signal_contract(s, "sr-done", {G_TYPE_POINTER, G_TYPE_POINTER});
    signal_contract(s, "start-sr", {G_TYPE_POINTER, G_TYPE_UINT, G_TYPE_UINT, G_TYPE_POINTER});
    signal_contract(s, "stop-sr", {G_TYPE_UINT});
    g_object_set(s, "select-rtp-protocol", 4, "latency", 200u,
        "init-rtsp-reconnect-interval", 5u, "rtsp-reconnect-interval", 5u,
        "smart-record", 2, "smart-rec-cache", m.config.record_cache_seconds,
        "smart-rec-container", 0, "smart-rec-mode", 1,
        "smart-rec-dir-path", m.record_directory.c_str(), "smart-rec-file-prefix", source.prefix.c_str(),
        "smart-rec-default-duration", 20u, nullptr);
    exact_numbers(s, {{"smart-record", 2}, {"smart-rec-cache", m.config.record_cache_seconds},
                     {"smart-rec-container", 0}, {"smart-rec-mode", 1}});
    source.record_signal = connect_callback(m, s, "sr-done", G_CALLBACK(record_done), source.index);
    require(source.record_signal != 0, SEEON_MEDIA_ERROR_PLUGIN);
  } else {
    properties(s, {"smart-record"});
    g_object_set(s, "smart-record", 0, nullptr);
  }
  exact_numbers(s, {{"type", source.rtsp ? 2u : 1u}, {"source-id", source.index},
                   {"disable-audio", 1}, {"cudadec-memtype", 0}, {"gpu-id", 0}});
  const auto pad_name = "sink_" + std::to_string(source.index);
  source.mux_pad = gst_element_request_pad_simple(m.mux, pad_name.c_str());
  require(source.mux_pad != nullptr, SEEON_MEDIA_ERROR_LINK);
  source.pad_signal = connect_callback(m, s, "pad-added", G_CALLBACK(pad_added), source.index);
  source.removed_signal = connect_callback(m, s, "pad-removed", G_CALLBACK(pad_removed), source.index);
  require(source.pad_signal && source.removed_signal, SEEON_MEDIA_ERROR_PLUGIN);
}
void build_preview(SeeonMedia &m) {
  auto &p = m.preview;
  p.queue = element(m.pipeline, "queue", "preview-queue");
  queue_limit(m, p.queue, true);
  p.valve = element(m.pipeline, "valve", "preview-valve");
  properties(p.valve, {"drop", "drop-mode"});
  g_object_set(p.valve, "drop", TRUE, "drop-mode", 2, nullptr);
  p.bin = gst_bin_new("preview-processing");
  require(p.bin != nullptr, SEEON_MEDIA_ERROR_PLUGIN);
  if (!gst_bin_add(GST_BIN(m.pipeline), p.bin)) {
    gst_object_unref(p.bin); p.bin = nullptr; throw Failure{SEEON_MEDIA_ERROR_PLUGIN};
  }
  p.tiler = element(p.bin, "nvmultistreamtiler", "preview-tiler");
  properties(p.tiler, {"rows", "columns", "width", "height", "show-source", "gpu-id"});
  g_object_set(p.tiler, "rows", 1u, "columns", 1u, "width", m.config.mux_width,
               "height", m.config.mux_height, "show-source", 0, "gpu-id", 0u, nullptr);
  exact_numbers(p.tiler, {{"width", m.config.mux_width}, {"height", m.config.mux_height},
                         {"rows", 1}, {"columns", 1}, {"gpu-id", 0}});
  auto *convert = element(p.bin, "nvvideoconvert", "preview-convert");
  p.osd = element(p.bin, "nvdsosd", "preview-osd");
  properties(p.osd, {"gpu-id", "process-mode", "display-bbox", "display-text"});
  g_object_set(p.osd, "gpu-id", 0u, "process-mode", 1, "display-bbox", TRUE, "display-text", TRUE, nullptr);
  auto *post = element(p.bin, "nvvideoconvert", "preview-post-convert");
  for (auto *convert_element : {convert, post}) {
    properties(convert_element, {"gpu-id", "compute-hw"});
    g_object_set(convert_element, "gpu-id", 0u, "compute-hw", 1, nullptr);
  }
  auto *filter = element(p.bin, "capsfilter", "preview-caps");
  auto *caps = gst_caps_from_string("video/x-raw(memory:NVMM),format=I420");
  require(caps != nullptr, SEEON_MEDIA_ERROR_PLUGIN);
  g_object_set(filter, "caps", caps, nullptr);
  gst_caps_unref(caps);
  auto *encoder = element(p.bin, "nvjpegenc", "preview-jpeg");
  p.sink = element(p.bin, "appsink", "preview-sink");
  g_object_set(p.sink, "emit-signals", TRUE, "max-buffers", 1u, "drop", TRUE,
               "sync", FALSE, "async", FALSE, "wait-on-eos", FALSE, "enable-last-sample", FALSE, nullptr);
  require(gst_element_link_many(p.tiler, convert, p.osd, post, filter, encoder, p.sink, nullptr),
          SEEON_MEDIA_ERROR_LINK);
  p.input_pad = gst_element_get_static_pad(p.tiler, "sink");
  require(p.input_pad != nullptr, SEEON_MEDIA_ERROR_LINK);
  auto *ghost = gst_ghost_pad_new("sink", p.input_pad);
  require(ghost != nullptr, SEEON_MEDIA_ERROR_LINK);
  if (!gst_element_add_pad(p.bin, ghost)) { gst_object_unref(ghost); throw Failure{SEEON_MEDIA_ERROR_LINK}; }
  require(gst_element_link_many(p.queue, p.valve, p.bin, nullptr), SEEON_MEDIA_ERROR_LINK);
  tee_link(m, p.queue, 1);
  p.gate_pad = gst_element_get_static_pad(p.valve, "src");
  require(p.gate_pad != nullptr, SEEON_MEDIA_ERROR_LINK);
  p.input_probe = add_probe(m, p.input_pad, GST_PAD_PROBE_TYPE_BUFFER, preview_input);
  p.sample_signal = connect_callback(m, p.sink, "new-sample", G_CALLBACK(preview_sample),
                                     SEEON_MEDIA_MAX_SOURCES, &p.sample_registration);
  require(p.input_probe && p.sample_signal, SEEON_MEDIA_ERROR_PLUGIN);
}
uint32_t error_domain(GQuark domain) noexcept {
  if (domain == GST_CORE_ERROR) return 1;
  if (domain == GST_LIBRARY_ERROR) return 2;
  if (domain == GST_RESOURCE_ERROR) return 3;
  if (domain == GST_STREAM_ERROR) return 4;
  return 0;
}
GstBusSyncReply bus_message(GstBus *, GstMessage *message, gpointer data) noexcept {
  CallbackGuard guard(data);
  if (!guard) {
    gst_message_unref(message);
    return GST_BUS_DROP;
  }
  auto &m = guard.media();
  try {
    if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR || GST_MESSAGE_TYPE(message) == GST_MESSAGE_WARNING) {
      GError *error = nullptr;
      const bool fatal = GST_MESSAGE_TYPE(message) == GST_MESSAGE_ERROR;
      if (fatal) gst_message_parse_error(message, &error, nullptr);
      else gst_message_parse_warning(message, &error, nullptr);
      const auto domain = error ? error_domain(error->domain) : 0;
      const auto code = error ? error->code : 0;
      if (error) g_error_free(error);
      if (fatal) m.fail(SEEON_MEDIA_ERROR_BUS, domain, code);
      else m.warn(SEEON_MEDIA_ERROR_BUS, domain, code);
    } else if (GST_MESSAGE_TYPE(message) == GST_MESSAGE_EOS && !m.stop_requested.load()) {
      m.fail(SEEON_MEDIA_ERROR_EOS);
    }
  } catch (...) { m.fail(SEEON_MEDIA_ERROR_EXCEPTION); }
  // The sync handler is the bus pump: retain no unbounded SDK message queue.
  gst_message_unref(message);
  return GST_BUS_DROP;
}
void build(SeeonMedia &m) {
  m.pipeline = gst_pipeline_new("seeon-native-media");
  require(m.pipeline != nullptr, SEEON_MEDIA_ERROR_PLUGIN);
  m.bus = gst_element_get_bus(m.pipeline);
  require(m.bus != nullptr, SEEON_MEDIA_ERROR_PLUGIN);
  // DeepStream's GStreamer has refcounted sync-handler registrations; notify
  // follows the last in-progress dispatch, including dispatch before entry.
  gst_bus_set_sync_handler(m.bus, bus_message,
                           callback_data(m, SEEON_MEDIA_MAX_SOURCES), callback_destroy);
  m.mux = element(m.pipeline, "nvstreammux", "mux");
  properties(m.mux, {"width", "height", "batch-size", "batched-push-timeout", "live-source", "gpu-id"});
  g_object_set(m.mux, "width", m.config.mux_width, "height", m.config.mux_height,
      "batch-size", m.config.source_count, "batched-push-timeout", int(m.config.mux_batch_timeout_us),
      "live-source", gboolean(m.config.mux_live_source), "gpu-id", 0u, nullptr);
  exact_numbers(m.mux, {{"width", m.config.mux_width}, {"height", m.config.mux_height},
      {"batch-size", m.config.source_count}, {"batched-push-timeout", m.config.mux_batch_timeout_us},
      {"live-source", m.config.mux_live_source}, {"gpu-id", 0}});
  auto *infer = element(m.pipeline, "nvinfer", "pose-infer");
  properties(infer, {"config-file-path"});
  g_object_set(infer, "config-file-path", m.infer_path.c_str(), nullptr);
  auto *tracker = element(m.pipeline, "nvtracker", "tracker");
  properties(tracker, {"ll-config-file", "ll-lib-file", "tracker-width", "tracker-height", "gpu-id"});
  g_object_set(tracker, "ll-config-file", m.tracker_config.c_str(), "ll-lib-file", m.tracker_library.c_str(),
      "tracker-width", m.config.tracker_width, "tracker-height", m.config.tracker_height, "gpu-id", 0u, nullptr);
  exact_numbers(tracker, {{"tracker-width", m.config.tracker_width}, {"tracker-height", m.config.tracker_height},
                         {"gpu-id", 0}});
  m.tee = element(m.pipeline, "tee", "media-tee");
  auto *queue = element(m.pipeline, "queue", "discard-queue");
  auto *sink = element(m.pipeline, "fakesink", "discard");
  queue_limit(m, queue, false);
  g_object_set(sink, "sync", FALSE, "async", FALSE, "enable-last-sample", FALSE, nullptr);
  require(gst_element_link_many(m.mux, infer, tracker, m.tee, nullptr) &&
          gst_element_link(queue, sink), SEEON_MEDIA_ERROR_LINK);
  tee_link(m, queue, 0);
  require(publication_meta_type() >= NVDS_START_USER_META, SEEON_MEDIA_ERROR_METADATA);
  m.metadata_pad = gst_element_get_static_pad(tracker, "src");
  require(m.metadata_pad != nullptr, SEEON_MEDIA_ERROR_LINK);
  // Publish and attach provenance on this same frame before tee/preview-queue.
  m.metadata_probe = add_probe(m, m.metadata_pad, GST_PAD_PROBE_TYPE_BUFFER, metadata_probe);
  require(m.metadata_probe != 0, SEEON_MEDIA_ERROR_PLUGIN);
  if (m.config.preview_enabled) build_preview(m);
  for (uint32_t i = 0; i < m.config.source_count; ++i) build_source(m, m.sources[i]);
}
void *control_main(void *data) noexcept {
  auto *m = static_cast<SeeonMedia *>(data);
  // Smart Record attaches its duration timers to GLib's process-default context.
  // Polling our controls and the synchronous bus alone never dispatches them.
  // The shared media graph owns that context; an unrelated dispatcher or second
  // graph must not silently steal its callbacks.
  struct ContextLease {
    GMainContext *context = g_main_context_ref(g_main_context_default());
    bool acquired = g_main_context_acquire(context) != FALSE;
    ~ContextLease() {
      if (acquired) g_main_context_release(context);
      g_main_context_unref(context);
    }
  } context;
  if (!context.acquired) m->fail(SEEON_MEDIA_ERROR_STATE);
  bool start_sent = false;
  while (!m->stop_requested.load()) {
    try {
      if (context.acquired) {
        for (unsigned dispatched = 0; dispatched < 16; ++dispatched) {
          if (!g_main_context_iteration(context.context, FALSE)) break;
        }
      }
      if (m->state.load() == SEEON_MEDIA_STARTING && !start_sent && !m->fatal.load()) {
        start_sent = true;
        require(gst_element_set_state(m->pipeline, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE,
                SEEON_MEDIA_ERROR_STATE);
      }
      if (start_sent && m->state.load() == SEEON_MEDIA_STARTING) {
        GstState current = GST_STATE_NULL;
        require(gst_element_get_state(m->pipeline, &current, nullptr, 0) != GST_STATE_CHANGE_FAILURE,
                SEEON_MEDIA_ERROR_STATE);
        if (current == GST_STATE_PLAYING && !m->stop_requested.load() && !m->fatal.load()) {
          m->state.store(SEEON_MEDIA_RUNNING);
          m->admitting.store(true);
          if (m->stop_requested.load() || m->fatal.load()) m->admitting.store(false);
        }
      }
      recording_tick(*m);
      if (m->config.preview_enabled) preview_tick(*m);
    } catch (const Failure &failure) { m->fail(failure.code); }
    catch (...) { m->fail(SEEON_MEDIA_ERROR_EXCEPTION); }
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  }
  m->admitting.store(false);
  m->state.store(SEEON_MEDIA_STOPPING);
  try {
    // Close entry before shutdown. NULL may be needed to unblock an already
    // leased vendor call; retain its owner throughout, then disconnect and await
    // every producer's finalizer. NULL/zero counters alone do not close entry.
    m->lifetime->entries.retire();
    auto &p = m->preview;
    if (p.valve) g_object_set(p.valve, "drop", TRUE, nullptr);
    if (p.gate_probe) { gst_pad_remove_probe(p.gate_pad, p.gate_probe); p.gate_probe = 0; }
    if (gst_element_set_state(m->pipeline, GST_STATE_NULL) == GST_STATE_CHANGE_FAILURE)
      m->fail(SEEON_MEDIA_ERROR_STATE);
    // set_state can block in a vendor task. It runs here, never on the caller's
    // deadline thread. Timeout keeps this thread and all callback storage owned.
    for (;;) {
      GstState current = GST_STATE_VOID_PENDING, pending = GST_STATE_VOID_PENDING;
      const auto change = gst_element_get_state(m->pipeline, &current, &pending, 0);
      if (change == GST_STATE_CHANGE_SUCCESS && current == GST_STATE_NULL &&
          pending == GST_STATE_VOID_PENDING) break;
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    disconnect(*m);
    recording_cancel(*m);
    preview_cancel(*m);
    release_graph(*m);
    m->state.store(SEEON_MEDIA_STOPPED);
    m->exited.store(true);
  } catch (...) {
    // Deliberately not exited/stopped: destroy must not free a possibly live SDK.
    m->fail(SEEON_MEDIA_ERROR_EXCEPTION);
  }
  return nullptr;
}
} // namespace

namespace seeon_media {
gulong connect_callback(SeeonMedia &m, GstElement *element, const char *name,
                        GCallback callback, uint32_t source, CallbackData *registration) {
  auto *data = callback_data(m, source, registration);
  auto *closure = g_cclosure_new(callback, data, closure_destroy);
  if (!closure) { callback_destroy(data); throw Failure{SEEON_MEDIA_ERROR_PLUGIN}; }
  // Hold a non-floating local reference, including the failed-connect path.
  g_closure_ref(closure);
  g_closure_sink(closure);
  const auto signal = g_signal_connect_closure(element, name, closure, FALSE);
  g_closure_unref(closure);
  return signal;
}
gulong add_probe(SeeonMedia &m, GstPad *pad, GstPadProbeType type,
                 GstPadProbeCallback callback, uint32_t source, CallbackData *registration) {
  require(pad != nullptr && GST_IS_PAD(pad), SEEON_MEDIA_ERROR_LINK);
  // All our probes return OK/DROP, never REMOVE during synchronous IDLE entry.
  // GstPad's probe-hook reference owns data through an already-dispatched call.
  return gst_pad_add_probe(pad, type, callback, callback_data(m, source, registration),
                           callback_destroy);
}
void disconnect(SeeonMedia &m) noexcept {
  m.lifetime->entries.retire();
  // A pad-added lease may still be installing its caps probe. Finish that work
  // before enumerating producers; later dispatches cannot touch the owner.
  while (!m.lifetime->entries.drained())
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  if (m.bus) {
    gst_bus_set_flushing(m.bus, TRUE);
    gst_bus_set_sync_handler(m.bus, nullptr, nullptr, nullptr);
  }
  if (m.metadata_probe) { gst_pad_remove_probe(m.metadata_pad, m.metadata_probe); m.metadata_probe = 0; }
  auto &p = m.preview;
  if (p.gate_probe) { gst_pad_remove_probe(p.gate_pad, p.gate_probe); p.gate_probe = 0; }
  if (p.input_probe) { gst_pad_remove_probe(p.input_pad, p.input_probe); p.input_probe = 0; }
  if (p.sample_signal) { g_signal_handler_disconnect(p.sink, p.sample_signal); p.sample_signal = 0; }
  for (auto &s : m.sources) {
    if (s.video_probe) { gst_pad_remove_probe(s.video_pad.load(), s.video_probe); s.video_probe = 0; }
    if (s.pad_signal) { g_signal_handler_disconnect(s.element, s.pad_signal); s.pad_signal = 0; }
    if (s.removed_signal) { g_signal_handler_disconnect(s.element, s.removed_signal); s.removed_signal = 0; }
    if (s.record_signal) { g_signal_handler_disconnect(s.element, s.record_signal); s.record_signal = 0; }
  }
  while (m.lifetime->registrations.load() != 0 || !m.lifetime->entries.drained())
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  p.gate_registration.reset();
  p.sample_registration.reset();
}
void release_graph(SeeonMedia &m) noexcept {
  disconnect(m);
  if (auto *sample = m.preview.sample.take()) gst_sample_unref(sample);
  for (auto &s : m.sources) {
    if (auto *pad = s.video_pad.exchange(nullptr)) gst_object_unref(pad);
    if (s.mux_pad) { gst_element_release_request_pad(m.mux, s.mux_pad); gst_object_unref(s.mux_pad); s.mux_pad = nullptr; }
    s.linked.store(false);
  }
  for (auto *&pad : m.tee_pads) {
    if (pad) { gst_element_release_request_pad(m.tee, pad); gst_object_unref(pad); pad = nullptr; }
  }
  for (auto **pad : {&m.metadata_pad, &m.preview.gate_pad, &m.preview.input_pad}) {
    if (*pad) { gst_object_unref(*pad); *pad = nullptr; }
  }
  if (m.bus) { gst_object_unref(m.bus); m.bus = nullptr; }
  if (m.pipeline) { gst_object_unref(m.pipeline); m.pipeline = nullptr; }
}
} // namespace seeon_media

extern "C" SeeonMediaResult seeon_media_open(const SeeonMediaConfig *config, SeeonMedia **out,
                                             SeeonMediaDiagnostic *error) {
  if (error) *error = {};
  if (!out) return SEEON_MEDIA_FATAL;
  *out = nullptr;
  SeeonMedia *m = nullptr;
  SeeonMediaError failure = SEEON_MEDIA_ERROR_EXCEPTION;
  try {
    require(config != nullptr, SEEON_MEDIA_ERROR_CONFIG);
    GError *init_error = nullptr;
    const bool initialized = gst_init_check(nullptr, nullptr, &init_error);
    if (init_error) g_error_free(init_error);
    require(initialized, SEEON_MEDIA_ERROR_PLUGIN);
    m = new SeeonMedia;
    copy_config(*m, *config);
    build(*m);
    require(!m->fatal.load(), SeeonMediaError(unpack(m->fatal.load()).code));
    require(pthread_create(&m->control, nullptr, control_main, m) == 0, SEEON_MEDIA_ERROR_STATE);
    *out = m;
    return SEEON_MEDIA_OK;
  } catch (const Failure &caught) { failure = caught.code; }
  catch (...) { failure = SEEON_MEDIA_ERROR_EXCEPTION; }
  const auto reported = unpack(m && m->fatal.load() ? m->fatal.load() : diagnostic(failure, 2));
  if (m) { release_graph(*m); delete m; }
  if (error) *error = reported;
  return SEEON_MEDIA_FATAL;
}
extern "C" SeeonMediaResult seeon_media_start(SeeonMedia *m) {
  return api(m, [&]() -> SeeonMediaResult {
    if (m->fatal.load()) return SEEON_MEDIA_FATAL;
    auto expected = SEEON_MEDIA_OPEN;
    if (m->stop_requested.load()) return SEEON_MEDIA_STALE;
    if (m->state.compare_exchange_strong(expected, SEEON_MEDIA_STARTING)) return SEEON_MEDIA_OK;
    return expected == SEEON_MEDIA_RUNNING ? SEEON_MEDIA_OK : SEEON_MEDIA_BUSY;
  });
}
extern "C" SeeonMediaResult seeon_media_read_status(SeeonMedia *m, SeeonMediaStatus *out, size_t bytes) {
  return api(m, [&]() -> SeeonMediaResult {
    if (!out || bytes < sizeof(*out)) return SEEON_MEDIA_TOO_SMALL;
    *out = {};
    out->state = m->state.load(); out->fatal = unpack(m->fatal.load()); out->warning = unpack(m->warning.load());
    out->warnings = m->warnings.load(); out->capacity_refusals = m->capacity_refusals.load();
    out->preview_dropped = m->preview_dropped.load(); out->late_record_callbacks = m->late_record_callbacks.load();
    out->source_count = m->config.source_count;
    out->callbacks_active = uint32_t(m->lifetime->entries.state.load() & ~LeaseGate::retired);
    out->stop_timed_out = m->stop_timed_out.load();
    std::lock_guard<std::mutex> lock(m->records_mutex);
    out->records_reserved = m->records_used.load();
    for (uint32_t i = 0; i < out->source_count; ++i) {
      auto &s = m->sources[i]; auto &target = out->sources[i];
      target.binding = s.binding; target.frames = s.frames.load(); target.overwritten = s.overwritten.load();
      target.dropped = s.dropped.load(); target.malformed = s.malformed.load();
      target.tensor_absent = s.tensor_absent.load(); target.objects = s.objects.load();
      target.video_linked = s.linked.load();
      const int active = s.active_record.load();
      if (active >= 0) target.active_record = ticket(m->records[active]);
    }
    return out->fatal.code ? SEEON_MEDIA_FATAL : SEEON_MEDIA_OK;
  });
}
extern "C" SeeonMediaResult seeon_media_stop(SeeonMedia *m, uint32_t deadline_ms) {
  return api(m, [&]() -> SeeonMediaResult {
    const auto deadline = Clock::now() + std::chrono::milliseconds(deadline_ms);
    m->admitting.store(false); m->stop_requested.store(true);
    while (!m->control_joined) {
      if (m->exited.load()) {
        // A completion flag alone does not prove thread exit: vendor TLS
        // destructors may still be running. Never perform an unbounded join.
        const int joined = pthread_tryjoin_np(m->control, nullptr);
        if (joined == 0) { m->control_joined = true; break; }
        if (joined != EBUSY) { m->fail(SEEON_MEDIA_ERROR_STATE); return SEEON_MEDIA_FATAL; }
      }
      if (Clock::now() >= deadline) {
        m->stop_timed_out.store(true); m->fail(SEEON_MEDIA_ERROR_STOP_TIMEOUT);
        return SEEON_MEDIA_FATAL;
      }
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    return SEEON_MEDIA_OK;
  });
}
extern "C" SeeonMediaResult seeon_media_destroy(SeeonMedia *m) {
  if (!m) return SEEON_MEDIA_FATAL;
  try {
    if (!m->control_joined || !m->exited.load() ||
        m->state.load() != SEEON_MEDIA_STOPPED || !m->lifetime->entries.drained() ||
        m->lifetime->registrations.load() != 0)
      return SEEON_MEDIA_BUSY;
    delete m;
    return SEEON_MEDIA_OK;
  } catch (...) { return SEEON_MEDIA_FATAL; }
}
