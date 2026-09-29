#include "media_internal.h"
#include <cmath>
#include <new>

using namespace seeon_media;
namespace {
// DeepStream's documented NvDsUserMeta protocol (DS_plugin_metadata.html):
// acquire from the batch pool, attach to the frame, and provide copy/release
// callbacks whose data argument is NvDsUserMeta, not the payload. Each SDK copy
// owns a separate immutable value; no callback retains a Source/SeeonMedia.
struct Publication {
  const SeeonMediaFrameIdentity frame;
};
gpointer copy_publication(gpointer data, gpointer) noexcept {
  const auto *user = static_cast<const NvDsUserMeta *>(data);
  if (!user || !user->user_meta_data) return nullptr;
  // Allocation failure leaves the copied frame unmarked/null, never fresh.
  return new (std::nothrow) Publication(*static_cast<const Publication *>(user->user_meta_data));
}
void release_publication(gpointer data, gpointer) noexcept {
  auto *user = static_cast<NvDsUserMeta *>(data);
  if (!user) return;
  delete static_cast<Publication *>(user->user_meta_data);
  user->user_meta_data = nullptr;
}
bool find_publication(const NvDsFrameMeta &frame, const NvDsUserMeta *&marker,
                      uint32_t &count) noexcept {
  marker = nullptr; count = 0;
  const auto type = publication_meta_type();
  if (type < NVDS_START_USER_META) return false;
  for (auto *node = frame.frame_user_meta_list; node; node = node->next) {
    if (++count > kMaxUserMeta || !node->data) return false;
    const auto *user = static_cast<const NvDsUserMeta *>(node->data);
    if (user->base_meta.meta_type != type) continue;
    if (marker) return false; // Even identical duplicates are ambiguous.
    marker = user;
  }
  return true;
}
bool attach_publication(NvDsBatchMeta &batch, NvDsFrameMeta &frame,
                        const SeeonMediaFrameIdentity &observed) noexcept {
  const NvDsUserMeta *existing = nullptr;
  uint32_t count = 0;
  // Single tracker streaming task, before tee: never restamp an old marker or
  // grow beyond the bounded user-meta traversal used by preview admission.
  if (!find_publication(frame, existing, count) || existing || count >= kMaxUserMeta) return false;
  std::unique_ptr<Publication> payload(new (std::nothrow) Publication{observed});
  if (!payload) return false;
  nvds_acquire_meta_lock(&batch);
  auto *user = nvds_acquire_user_meta_from_pool(&batch);
  if (user) {
    user->user_meta_data = payload.release();
    user->base_meta.meta_type = publication_meta_type();
    user->base_meta.uContext = nullptr;
    user->base_meta.copy_func = copy_publication;
    user->base_meta.release_func = release_publication;
    nvds_add_user_meta_to_frame(&frame, user);
  }
  nvds_release_meta_lock(&batch);
  return user != nullptr; // Pool failure does not publish a pose or advance frames.
}
bool identity(SeeonMedia &m, const NvDsFrameMeta &f, uint64_t sequence,
              SeeonMediaFrameIdentity &out) noexcept {
  if (f.pad_index >= m.config.source_count || f.source_id != f.pad_index ||
      f.batch_id >= m.config.source_count || f.num_surfaces_per_frame != 1 ||
      !f.source_frame_width || !f.source_frame_height ||
      (f.pipeline_width && f.pipeline_width != m.config.mux_width) ||
      (f.pipeline_height && f.pipeline_height != m.config.mux_height)) return false;
  out = {m.sources[f.pad_index].binding, sequence, f.buf_pts, f.frame_num,
         uint32_t(GST_CLOCK_TIME_IS_VALID(f.buf_pts)), f.source_id, f.batch_id, f.pad_index,
         f.source_frame_width, f.source_frame_height, m.config.mux_width, m.config.mux_height};
  return true;
}
bool published_identity(SeeonMedia &m, const NvDsFrameMeta &frame,
                        SeeonMediaFrameIdentity &out) noexcept {
  const NvDsUserMeta *marker = nullptr;
  uint32_t count = 0;
  if (!find_publication(frame, marker, count) || !marker || !marker->user_meta_data ||
      marker->base_meta.copy_func != copy_publication ||
      marker->base_meta.release_func != release_publication || marker->base_meta.uContext) return false;
  const auto &published = static_cast<const Publication *>(marker->user_meta_data)->frame;
  SeeonMediaFrameIdentity observed{};
  if (!published.sequence || !identity(m, frame, published.sequence, observed) ||
      !same_binding(published.binding, observed.binding) ||
      published.source_id != observed.source_id || published.pad_index != observed.pad_index ||
      published.batch_id != observed.batch_id || published.pts_ns != observed.pts_ns ||
      published.pts_valid != observed.pts_valid || published.frame_number != observed.frame_number ||
      published.source_width != observed.source_width || published.source_height != observed.source_height ||
      published.analysis_width != observed.analysis_width || published.analysis_height != observed.analysis_height)
    return false;
  out = published; // Never reconstruct provenance from the current source counter.
  return true;
}
bool valid_batch(const SeeonMedia &m, const NvDsBatchMeta *batch) noexcept {
  if (!batch || batch->num_frames_in_batch > m.config.source_count ||
      batch->max_frames_in_batch > SEEON_MEDIA_MAX_SOURCES ||
      batch->num_frames_in_batch > batch->max_frames_in_batch) return false;
  uint32_t count = 0, pads = 0;
  for (auto *node = batch->frame_meta_list; node; node = node->next) {
    if (++count > batch->num_frames_in_batch || !node->data) return false;
    auto &f = *static_cast<const NvDsFrameMeta *>(node->data);
    if (f.pad_index >= m.config.source_count || f.batch_id >= batch->max_frames_in_batch ||
        (pads & (1u << f.pad_index))) return false;
    pads |= 1u << f.pad_index;
  }
  return count == batch->num_frames_in_batch;
}
bool copy_tensor(const NvDsFrameMeta &frame, SeeonMediaPose &out) noexcept {
  uint32_t meta_count = 0;
  for (auto *node = frame.frame_user_meta_list; node; node = node->next) {
    if (++meta_count > kMaxUserMeta || !node->data) return false;
    auto &user = *static_cast<const NvDsUserMeta *>(node->data);
    if (user.base_meta.meta_type != NVDSINFER_TENSOR_OUTPUT_META) continue;
    auto *tensor = static_cast<const NvDsInferTensorMeta *>(user.user_meta_data);
    if (!tensor) return false;
    if (tensor->unique_id != 1) continue;
    if (out.tensor_present || !tensor->output_layers_info || !tensor->out_buf_ptrs_host ||
        !tensor->num_output_layers || tensor->num_output_layers > kMaxLayers ||
        tensor->network_info.width != 640 || tensor->network_info.height != 640 ||
        !tensor->maintain_aspect_ratio || tensor->symmetric_padding) return false;
    bool found = false;
    for (uint32_t index = 0; index < tensor->num_output_layers; ++index) {
      const auto &layer = tensor->output_layers_info[index];
      if (!layer.layerName || std::strncmp(layer.layerName, "output0", 8) != 0) continue;
      const auto &dims = layer.inferDims;
      const bool shape = (dims.numDims == 2 && dims.d[0] == 300 && dims.d[1] == 57) ||
          (dims.numDims == 3 && dims.d[0] == 1 && dims.d[1] == 300 && dims.d[2] == 57);
      if (found || layer.isInput || layer.dataType != FLOAT || !shape ||
          dims.numElements != 300 * 57 || !tensor->out_buf_ptrs_host[index]) return false;
      // layer.buffer is NOT valid in NvDsInferTensorMeta. Host buffers are
      // indexed by output-layer position, not TensorRT bindingIndex/batch_id.
      std::memcpy(out.rows, tensor->out_buf_ptrs_host[index], sizeof(out.rows));
      for (const auto &row : out.rows)
        for (float value : row) if (!std::isfinite(value)) return false;
      found = true;
    }
    if (!found) return false;
    out.tensor_present = 1; out.row_count = SEEON_MEDIA_POSE_ROWS;
  }
  return true; // No tensor: retain real tracked objects with row_count=0.
}
bool copy_objects(const NvDsFrameMeta &frame, SeeonMediaPose &out) noexcept {
  if (frame.num_obj_meta > SEEON_MEDIA_MAX_OBJECTS) return false;
  for (auto *node = frame.obj_meta_list; node; node = node->next) {
    if (!node->data || out.object_count >= SEEON_MEDIA_MAX_OBJECTS) return false;
    const auto &object = *static_cast<const NvDsObjectMeta *>(node->data);
    const auto &r = object.rect_params;
    if (!std::isfinite(r.left) || !std::isfinite(r.top) || !std::isfinite(r.width) ||
        !std::isfinite(r.height) || !std::isfinite(object.confidence)) return false;
    out.objects[out.object_count++] = {object.object_id, r.left, r.top, r.width, r.height, object.confidence};
  }
  return out.object_count == frame.num_obj_meta;
}
void malformed(SeeonMedia &m, Source &s) noexcept {
  s.malformed.fetch_add(1); s.dropped.fetch_add(1);
  if (++s.consecutive_failures >= kProbeFailureThreshold) m.fail(SEEON_MEDIA_ERROR_METADATA);
}
void preview_failed(Preview &p, SeeonMediaResult result, SeeonMediaError error) noexcept {
  p.result.result = result; p.result.error = error; p.result.jpeg_bytes = 0;
  p.phase.store(PreviewPhase::Ready);
}
void collect_preview(SeeonMedia &m) noexcept {
  auto &p = m.preview;
  Clock::time_point completed{};
  std::unique_ptr<GstSample, decltype(&gst_sample_unref)> sample(p.sample.take(&completed), &gst_sample_unref);
  if (!sample) return;
  if (!p.encoder_pending) { m.preview_dropped.fetch_add(1); return; }
  auto *buffer = gst_sample_get_buffer(sample.get());
  if (!buffer || !GST_BUFFER_PTS_IS_VALID(buffer) || GST_BUFFER_PTS(buffer) != p.result.batch_pts_ns) {
    if (p.phase.load() == PreviewPhase::Inflight)
      preview_failed(p, SEEON_MEDIA_STALE, SEEON_MEDIA_ERROR_PREVIEW_IDENTITY);
    else m.preview_dropped.fetch_add(1);
    return; // Wrong PTS is not retirement of the quarantined operation.
  }
  p.encoder_pending = false;
  // A timeout is immutable even if its result has already been consumed.
  if (p.phase.load() != PreviewPhase::Inflight) { m.preview_dropped.fetch_add(1); return; }
  if (completed >= p.deadline) {
    preview_failed(p, SEEON_MEDIA_FATAL, SEEON_MEDIA_ERROR_PREVIEW_TIMEOUT);
    return;
  }
  const auto size = gst_buffer_get_size(buffer);
  if (size < 4 || size > p.bytes.size()) {
    preview_failed(p, SEEON_MEDIA_TOO_SMALL, SEEON_MEDIA_ERROR_PREVIEW_SIZE);
    return;
  }
  if (gst_buffer_extract(buffer, 0, p.bytes.data(), size) != size ||
      p.bytes[0] != 0xff || p.bytes[1] != 0xd8 || p.bytes[size - 2] != 0xff || p.bytes[size - 1] != 0xd9) {
    preview_failed(p, SEEON_MEDIA_FATAL, SEEON_MEDIA_ERROR_PREVIEW_IDENTITY);
    return;
  }
  p.result.jpeg_bytes = size; p.result.result = SEEON_MEDIA_OK; p.result.error = SEEON_MEDIA_ERROR_NONE;
  p.phase.store(PreviewPhase::Ready);
}
void remove_gate(Preview &p) noexcept {
  if (p.gate_probe) {
    p.gate_registration->entries.retire();
    gst_pad_remove_probe(p.gate_pad, p.gate_probe); p.gate_probe = 0;
  }
  if (!p.gate_registration || p.gate_registration->detached.load()) {
    p.gate_registration.reset();
    p.control_busy.store(false);
  }
}
} // namespace

namespace seeon_media {
NvDsMetaType publication_meta_type() noexcept {
  static gchar descriptor[] = "SEEON.NATIVE.POSE_PUBLICATION";
  static const NvDsMetaType type = nvds_get_user_meta_type(descriptor);
  return type;
}
GstPadProbeReturn metadata_probe(GstPad *, GstPadProbeInfo *info, gpointer data) noexcept {
  CallbackGuard guard(data);
  if (!guard) return GST_PAD_PROBE_OK;
  auto &m = guard.media();
  try {
    if (admission(m) != SEEON_MEDIA_OK) return GST_PAD_PROBE_OK;
    auto *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    auto *batch = buffer ? gst_buffer_get_nvds_batch_meta(buffer) : nullptr;
    if (!valid_batch(m, batch)) { m.fail(SEEON_MEDIA_ERROR_METADATA); return GST_PAD_PROBE_OK; }
    for (auto *node = batch->frame_meta_list; node; node = node->next) {
      auto &frame = *static_cast<NvDsFrameMeta *>(node->data);
      auto &s = m.sources[frame.pad_index];
      SeeonMediaFrameIdentity observed{};
      if (!identity(m, frame, 0, observed)) { malformed(m, s); continue; }
      std::unique_lock<std::mutex> lock(s.pose_mutex, std::try_to_lock);
      if (!lock.owns_lock()) { s.dropped.fetch_add(1); continue; }
      const auto published = s.frames.load();
      if (published == UINT64_MAX) { m.fail(SEEON_MEDIA_ERROR_METADATA); return GST_PAD_PROBE_OK; }
      observed.sequence = published + 1;
      s.pose_staging = {};
      s.pose_staging.frame = observed;
      if (!copy_tensor(frame, s.pose_staging) || !copy_objects(frame, s.pose_staging) ||
          !attach_publication(*batch, frame, observed)) {
        malformed(m, s); continue;
      }
      s.consecutive_failures = 0;
      s.objects.fetch_add(s.pose_staging.object_count);
      if (!s.pose_staging.tensor_present) s.tensor_absent.fetch_add(1);
      if (s.pose_ready) s.overwritten.fetch_add(1);
      s.pose = s.pose_staging;
      s.pose_ready = true;
      // Publication linearizes here, while pose_mutex still hides the mailbox
      // and the SDK probe still holds this marked frame upstream of the queue.
      s.frames.fetch_add(1);
    }
  } catch (...) { m.fail(SEEON_MEDIA_ERROR_EXCEPTION); }
  return GST_PAD_PROBE_OK;
}
GstPadProbeReturn preview_idle(GstPad *, GstPadProbeInfo *, gpointer data) noexcept {
  CallbackGuard guard(data);
  if (!guard) return GST_PAD_PROBE_OK;
  guard.media().preview.gate_idle.store(true);
  return GST_PAD_PROBE_OK; // IDLE remains installed, blocking all new input.
}
GstPadProbeReturn preview_input(GstPad *, GstPadProbeInfo *info, gpointer data) noexcept {
  CallbackGuard guard(data);
  if (!guard) return GST_PAD_PROBE_DROP;
  auto &m = guard.media();
  PreviewGuard preview_guard(m.preview);
  try {
    auto &p = m.preview;
    if (admission(m) != SEEON_MEDIA_OK) return GST_PAD_PROBE_DROP;
    std::unique_lock<std::mutex> lock(p.mutex, std::try_to_lock);
    if (!lock.owns_lock()) { m.preview_dropped.fetch_add(1); return GST_PAD_PROBE_DROP; }
    if (p.phase.load() != PreviewPhase::Armed) return GST_PAD_PROBE_DROP;
    preview_expire(m);
    if (p.phase.load() != PreviewPhase::Armed) return GST_PAD_PROBE_DROP;
    auto *buffer = GST_PAD_PROBE_INFO_BUFFER(info);
    auto *batch = buffer ? gst_buffer_get_nvds_batch_meta(buffer) : nullptr;
    if (!valid_batch(m, batch) || !GST_BUFFER_PTS_IS_VALID(buffer)) {
      preview_failed(p, SEEON_MEDIA_STALE, SEEON_MEDIA_ERROR_PREVIEW_IDENTITY);
      return GST_PAD_PROBE_DROP;
    }
    for (auto *node = batch->frame_meta_list; node; node = node->next) {
      const auto &frame = *static_cast<const NvDsFrameMeta *>(node->data);
      if (frame.pad_index != p.source) continue;
      SeeonMediaFrameIdentity published{};
      if (!published_identity(m, frame, published) ||
          !same_binding(published.binding, p.result.frame.binding) ||
          published.sequence <= p.publication_floor) {
        m.preview_dropped.fetch_add(1);
        return GST_PAD_PROBE_DROP; // Missing, duplicate, mismatched or pre-fence; keep waiting.
      }
      // A batch lacking the selected source is never fed to the tiler's cache.
      if (!published.pts_valid || frame.frame_num < 0) {
        preview_failed(p, SEEON_MEDIA_STALE, SEEON_MEDIA_ERROR_PREVIEW_IDENTITY);
        return GST_PAD_PROBE_DROP;
      }
      p.result.frame = published;
      p.result.batch_pts_ns = GST_BUFFER_PTS(buffer);
      p.encoder_pending = true;
      p.phase.store(PreviewPhase::Inflight);
      return GST_PAD_PROBE_OK; // Exactly one batched buffer enters this request.
    }
  } catch (...) { m.fail(SEEON_MEDIA_ERROR_EXCEPTION); }
  return GST_PAD_PROBE_DROP;
}
GstFlowReturn preview_sample(GstAppSink *sink, gpointer data) noexcept {
  CallbackGuard guard(data);
  if (!guard) return GST_FLOW_OK;
  auto &m = guard.media();
  PreviewGuard preview_guard(m.preview);
  try {
    std::unique_ptr<GstSample, decltype(&gst_sample_unref)> sample(
        gst_app_sink_try_pull_sample(sink, 0), &gst_sample_unref);
    if (!sample) return GST_FLOW_OK;
    // One input was admitted. Transfer its sole owned sample without taking the
    // poll mutex, copying JPEG bytes, allocating, or waiting for the consumer.
    if (m.preview.sample.publish(sample.get(), Clock::now())) sample.release();
    else m.preview_dropped.fetch_add(1); // Only an extra sample can fill an occupied mailbox.
  } catch (...) { m.fail(SEEON_MEDIA_ERROR_EXCEPTION); return GST_FLOW_ERROR; }
  return GST_FLOW_OK;
}
void preview_expire(SeeonMedia &m) {
  auto &p = m.preview;
  const auto phase = p.phase.load();
  if (phase == PreviewPhase::Idle || phase == PreviewPhase::Ready) return;
  if (m.fatal.load()) preview_failed(p, SEEON_MEDIA_FATAL, SeeonMediaError(unpack(m.fatal.load()).code));
  else if (m.stop_requested.load()) preview_failed(p, SEEON_MEDIA_STALE, SEEON_MEDIA_ERROR_CANCELLED);
  else if (Clock::now() >= p.deadline) preview_failed(p, SEEON_MEDIA_FATAL, SEEON_MEDIA_ERROR_PREVIEW_TIMEOUT);
}
void preview_tick(SeeonMedia &m) {
  auto &p = m.preview;
  std::unique_lock<std::mutex> lock(p.mutex, std::try_to_lock);
  if (!lock.owns_lock()) return;
  collect_preview(m); // On-time completion wins over expiry delayed by polling.
  preview_expire(m);
  const auto phase = p.phase.load();
  if (p.control_busy.load() && !p.gate_probe) {
    // Removal may have returned before the final probe invocation released its
    // SDK reference. Do not accumulate registrations or start a new request.
    lock.unlock();
    remove_gate(p);
    return;
  }
  if (!p.control_busy.load()) {
    const bool preparing = phase == PreviewPhase::Queued;
    const bool quarantine = p.encoder_pending && phase != PreviewPhase::Inflight;
    if (!preparing && !quarantine) {
      lock.unlock();
      if (phase != PreviewPhase::Armed) g_object_set(p.valve, "drop", TRUE, nullptr);
      return;
    }
    p.control_busy.store(true);
    lock.unlock();
    // Initial negotiation must pass before installing an IDLE gate. Quarantine
    // does not need negotiation and must still drain on error/timeout.
    if (preparing) {
      for (auto type : {GST_EVENT_STREAM_START, GST_EVENT_CAPS, GST_EVENT_SEGMENT}) {
        auto *event = gst_pad_get_sticky_event(p.gate_pad, type, 0);
        if (!event) { p.control_busy.store(false); return; }
        gst_event_unref(event);
      }
    }
    lock.lock();
    preview_expire(m);
    if (p.phase.load() == PreviewPhase::Queued) p.phase.store(PreviewPhase::Draining);
    p.gate_idle.store(false);
    lock.unlock();
    g_object_set(p.valve, "drop", TRUE, nullptr);
    p.gate_probe = add_probe(m, p.gate_pad, GST_PAD_PROBE_TYPE_IDLE, preview_idle,
                             SEEON_MEDIA_MAX_SOURCES, &p.gate_registration);
    require(p.gate_probe != 0, SEEON_MEDIA_ERROR_STATE);
    return;
  }
  if (!p.gate_idle.load()) return;
  lock.unlock();

  // Retire the actual sample producer, including invocations dispatched before
  // guard entry. A zero p.callbacks snapshot alone is NOT an entry barrier.
  if (p.sample_signal) {
    p.sample_registration->entries.retire();
    g_signal_handler_disconnect(p.sink, p.sample_signal);
    p.sample_signal = 0;
  }
  // The blocked upstream gate + complete NULL transition retire the encoder and
  // tiler operation. No new request is admitted while this can block in the SDK.
  require(gst_element_set_state(p.bin, GST_STATE_NULL) != GST_STATE_CHANGE_FAILURE, SEEON_MEDIA_ERROR_STATE);
  GstState current = GST_STATE_VOID_PENDING, pending = GST_STATE_VOID_PENDING;
  const auto changed = gst_element_get_state(p.bin, &current, &pending, 0);
  require(changed != GST_STATE_CHANGE_FAILURE, SEEON_MEDIA_ERROR_STATE);
  if (changed != GST_STATE_CHANGE_SUCCESS || current != GST_STATE_NULL ||
      pending != GST_STATE_VOID_PENDING || p.callbacks.load() != 0 ||
      (p.sample_registration && !p.sample_registration->detached.load())) return;
  p.sample_registration.reset();
  if (auto *stale = gst_app_sink_try_pull_sample(GST_APP_SINK(p.sink), 0)) gst_sample_unref(stale);
  lock.lock();
  if (auto *stale = p.sample.take()) {
    gst_sample_unref(stale);
    m.preview_dropped.fetch_add(1);
  }
  p.encoder_pending = false;
  preview_expire(m);
  const bool preparing = p.phase.load() == PreviewPhase::Draining;
  const auto source = p.source;
  const auto draw = p.draw;
  lock.unlock();
  if (preparing) {
    g_object_set(p.tiler, "show-source", int(source), nullptr);
    g_object_set(p.osd, "display-bbox", gboolean(draw), "display-text", gboolean(draw), nullptr);
    p.sample_signal = connect_callback(m, p.sink, "new-sample", G_CALLBACK(preview_sample),
                                       SEEON_MEDIA_MAX_SOURCES, &p.sample_registration);
    require(p.sample_signal != 0, SEEON_MEDIA_ERROR_PLUGIN);
    require(gst_element_set_state(p.bin, GST_STATE_PLAYING) != GST_STATE_CHANGE_FAILURE, SEEON_MEDIA_ERROR_STATE);
    // Replay only the real sticky events, with buffers still blocked.
    std::unique_ptr<GstPad, decltype(&gst_object_unref)> sink(
        gst_element_get_static_pad(p.bin, "sink"), &gst_object_unref);
    require(sink != nullptr, SEEON_MEDIA_ERROR_STATE);
    for (auto type : {GST_EVENT_STREAM_START, GST_EVENT_CAPS, GST_EVENT_SEGMENT}) {
      auto *event = gst_pad_get_sticky_event(p.gate_pad, type, 0);
      require(event != nullptr, SEEON_MEDIA_ERROR_STATE);
      require(gst_pad_send_event(sink.get(), event), SEEON_MEDIA_ERROR_STATE);
    }
    lock.lock();
    preview_expire(m);
    const bool armed = p.phase.load() == PreviewPhase::Draining;
    if (armed) p.phase.store(PreviewPhase::Armed);
    lock.unlock();
    // Expiry may race this set, but preview_input rechecks phase under mutex.
    if (armed) g_object_set(p.valve, "drop", FALSE, nullptr);
  }
  remove_gate(p);
}
void preview_cancel(SeeonMedia &m) {
  if (!m.config.preview_enabled) return;
  std::lock_guard<std::mutex> lock(m.preview.mutex);
  preview_expire(m);
  if (auto *sample = m.preview.sample.take()) gst_sample_unref(sample);
  m.preview.encoder_pending = false;
  m.preview.control_busy.store(false);
}
} // namespace seeon_media

extern "C" SeeonMediaResult seeon_media_try_read_pose(SeeonMedia *m, uint32_t source,
                                                       SeeonMediaPose *out, size_t bytes) {
  return api(m, [&]() -> SeeonMediaResult {
    if (!out || bytes < sizeof(*out)) return SEEON_MEDIA_TOO_SMALL;
    if (source >= m->config.source_count) return SEEON_MEDIA_STALE;
    if (m->fatal.load()) return SEEON_MEDIA_FATAL;
    if (!m->admitting.load()) return SEEON_MEDIA_EMPTY;
    auto &s = m->sources[source];
    std::unique_lock<std::mutex> lock(s.pose_mutex, std::try_to_lock);
    if (!lock.owns_lock()) return SEEON_MEDIA_BUSY;
    if (!s.pose_ready) return SEEON_MEDIA_EMPTY;
    *out = s.pose; s.pose_ready = false;
    return SEEON_MEDIA_OK;
  });
}
extern "C" SeeonMediaResult seeon_media_request_preview(SeeonMedia *m, uint32_t source,
    const SeeonMediaBinding *binding, uint64_t request, uint32_t draw, uint32_t timeout_ms) {
  return api(m, [&]() -> SeeonMediaResult {
    auto status = source_check(*m, source, binding);
    if (status != SEEON_MEDIA_OK) return status;
    if (!m->config.preview_enabled) return SEEON_MEDIA_UNSUPPORTED;
    status = admission(*m);
    if (status != SEEON_MEDIA_OK) return status;
    if (!request || draw > 1 || !timeout_ms || timeout_ms > 60000) return SEEON_MEDIA_STALE;
    if (!m->sources[source].linked.load() || !m->sources[source].frames.load()) return SEEON_MEDIA_BUSY;
    auto &p = m->preview;
    std::unique_lock<std::mutex> lock(p.mutex, std::try_to_lock);
    if (!lock.owns_lock()) return SEEON_MEDIA_BUSY;
    if (request <= p.last_request) return SEEON_MEDIA_STALE;
    if (p.control_busy.load() || p.encoder_pending || p.phase.load() != PreviewPhase::Idle)
      return m->capacity();
    p.last_request = request; p.source = source; p.draw = draw != 0;
    // A frame still upstream that publishes after this load is on the new
    // logical side. This fence makes no claim about RTSP capture time.
    p.publication_floor = m->sources[source].frames.load();
    p.result = {}; p.result.request_id = request;
    p.result.frame.binding = *binding; p.result.frame.source_id = source;
    p.deadline = Clock::now() + std::chrono::milliseconds(timeout_ms);
    p.phase.store(PreviewPhase::Queued);
    return SEEON_MEDIA_OK;
  });
}
extern "C" SeeonMediaResult seeon_media_try_read_preview(SeeonMedia *m, SeeonMediaPreview *out,
    size_t descriptor_bytes, uint8_t *jpeg, size_t jpeg_capacity) {
  return api(m, [&]() -> SeeonMediaResult {
    if (!m->config.preview_enabled) return SEEON_MEDIA_UNSUPPORTED;
    if (!out || descriptor_bytes < sizeof(*out)) return SEEON_MEDIA_TOO_SMALL;
    auto &p = m->preview;
    std::unique_lock<std::mutex> lock(p.mutex, std::try_to_lock);
    if (!lock.owns_lock()) return SEEON_MEDIA_BUSY;
    collect_preview(*m);
    preview_expire(*m);
    if (p.phase.load() != PreviewPhase::Ready) return m->fatal.load() ? SEEON_MEDIA_FATAL : SEEON_MEDIA_EMPTY;
    *out = p.result;
    if (p.result.jpeg_bytes && (!jpeg || jpeg_capacity < p.result.jpeg_bytes)) return SEEON_MEDIA_TOO_SMALL;
    if (p.result.jpeg_bytes) std::memcpy(jpeg, p.bytes.data(), size_t(p.result.jpeg_bytes));
    p.phase.store(PreviewPhase::Idle);
    return SEEON_MEDIA_OK;
  });
}
