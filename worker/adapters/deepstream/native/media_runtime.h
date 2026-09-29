#ifndef SEEON_MEDIA_RUNTIME_H
#define SEEON_MEDIA_RUNTIME_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct SeeonMedia SeeonMedia;
#define SEEON_MEDIA_ABI_VERSION 1u
#define SEEON_MEDIA_MAX_SOURCES 16u
#define SEEON_MEDIA_POSE_ROWS 300u
#define SEEON_MEDIA_POSE_COLUMNS 57u
#define SEEON_MEDIA_MAX_OBJECTS 150u
#define SEEON_MEDIA_MAX_RECORDS 256u
#define SEEON_MEDIA_PATH_BYTES 4096u
#define SEEON_MEDIA_MAX_PREVIEW_BYTES (16u * 1024u * 1024u)

typedef enum SeeonMediaResult {
  SEEON_MEDIA_OK = 0, SEEON_MEDIA_EMPTY = 1, SEEON_MEDIA_BUSY = 2,
  SEEON_MEDIA_STALE = 3, SEEON_MEDIA_TOO_SMALL = 4,
  SEEON_MEDIA_UNSUPPORTED = 5, SEEON_MEDIA_FATAL = 6
} SeeonMediaResult;

typedef enum SeeonMediaError {
  SEEON_MEDIA_ERROR_NONE = 0, SEEON_MEDIA_ERROR_CONFIG = 1,
  SEEON_MEDIA_ERROR_PLUGIN = 2, SEEON_MEDIA_ERROR_LINK = 3,
  SEEON_MEDIA_ERROR_STATE = 4, SEEON_MEDIA_ERROR_BUS = 5,
  SEEON_MEDIA_ERROR_EOS = 6, SEEON_MEDIA_ERROR_METADATA = 7,
  SEEON_MEDIA_ERROR_SOURCE_CAPS = 8, SEEON_MEDIA_ERROR_EXCEPTION = 9,
  SEEON_MEDIA_ERROR_CAPACITY = 10, SEEON_MEDIA_ERROR_PREVIEW_TIMEOUT = 11,
  SEEON_MEDIA_ERROR_PREVIEW_IDENTITY = 12, SEEON_MEDIA_ERROR_PREVIEW_SIZE = 13,
  SEEON_MEDIA_ERROR_RECORD_START = 14, SEEON_MEDIA_ERROR_RECORD_IDENTITY = 15,
  SEEON_MEDIA_ERROR_RECORD_INFO = 16, SEEON_MEDIA_ERROR_RECORD_TIMEOUT = 17,
  SEEON_MEDIA_ERROR_STOP_TIMEOUT = 18, SEEON_MEDIA_ERROR_CANCELLED = 19
} SeeonMediaError;

typedef enum SeeonMediaState {
  SEEON_MEDIA_OPEN = 0, SEEON_MEDIA_STARTING = 1, SEEON_MEDIA_RUNNING = 2,
  SEEON_MEDIA_STOPPING = 3, SEEON_MEDIA_STOPPED = 4
} SeeonMediaState;

/* token identifies the complete Rust-owned SourceBinding (boot, child, camera,
 * transform). Neither token nor generation/epoch can change on this handle. */
typedef struct SeeonMediaBinding {
  uint64_t token, generation, epoch;
} SeeonMediaBinding;

typedef struct SeeonMediaSource {
  uint32_t source_id; /* Must equal this entry's immutable roster/mux-pad index. */
  SeeonMediaBinding binding;
  const char *uri;
  const char *record_prefix; /* Unique, nonempty safe filename component for RTSP. */
} SeeonMediaSource;

/* All pointers are borrowed for open only; strings and roster are copied.
 * Required numeric bounds have no implicit defaults. The admitted nvinfer file
 * must reference an existing engine and the existing Yolo26 parser, unique-id 1,
 * output0, tensor meta, and this exact batch. Model/builder inputs are refused:
 * this owner never allows nvinfer to build an engine as a runtime fallback.
 * Config/artifact immutability and preprocessing admission belong to the caller.
 * file:// is accepted only with allow_file_uris=1, and never records.
 * record_capacity bounds currently occupied reservations (1..256), not lifetime
 * requests. Reuse requires consumed delivery, a retired SDK operation, and
 * drained callback writers. Unread completions and unresolved failed operations
 * retain capacity. A timed-out source refuses successors until SDK retirement.
 */
typedef struct SeeonMediaConfig {
  uint32_t abi_version, struct_size, source_count;
  const SeeonMediaSource *sources;
  const char *infer_config_path, *tracker_config_path, *tracker_library_path;
  const char *record_directory;
  uint32_t record_cache_seconds, record_capacity;
  uint32_t mux_width, mux_height, mux_batch_timeout_us, mux_live_source;
  uint32_t tracker_width, tracker_height;
  uint32_t queue_max_buffers; /* 1..64; bytes/time queue limits are disabled. */
  uint32_t preview_enabled, max_preview_bytes, allow_file_uris;
} SeeonMediaConfig;

typedef struct SeeonMediaFrameIdentity {
  SeeonMediaBinding binding;
  /* sequence is the same-frame pose publication ordinal for both pose and
   * successful preview receipts, distinct from the SDK frame_number. */
  uint64_t sequence, pts_ns;
  int64_t frame_number;
  uint32_t pts_valid, source_id, batch_id, pad_index;
  uint32_t source_width, source_height, analysis_width, analysis_height;
} SeeonMediaFrameIdentity;

typedef struct SeeonMediaObject {
  uint64_t track_id; /* Includes the SDK's UINT64_MAX untracked sentinel. */
  float left, top, width, height, confidence;
} SeeonMediaObject;

typedef struct SeeonMediaPose {
  SeeonMediaFrameIdentity frame;
  uint32_t tensor_present, row_count, object_count;
  float rows[SEEON_MEDIA_POSE_ROWS][SEEON_MEDIA_POSE_COLUMNS];
  SeeonMediaObject objects[SEEON_MEDIA_MAX_OBJECTS];
} SeeonMediaPose;

typedef struct SeeonMediaPreview {
  uint64_t request_id, jpeg_bytes, batch_pts_ns;
  SeeonMediaFrameIdentity frame;
  SeeonMediaResult result;
  SeeonMediaError error;
} SeeonMediaPreview;

typedef struct SeeonMediaRecordTicket {
  SeeonMediaBinding binding;
  uint64_t request_id;
  uint32_t source_id, session_id, session_valid, coalesced;
} SeeonMediaRecordTicket;

typedef struct SeeonMediaRecord {
  SeeonMediaRecordTicket ticket;
  SeeonMediaResult result;
  SeeonMediaError error;
  uint64_t duration_ms;
  uint32_t width, height, contains_video, contains_audio;
  char directory[SEEON_MEDIA_PATH_BYTES];
  char filename[SEEON_MEDIA_PATH_BYTES];
} SeeonMediaRecord;

/* Sanitized diagnostics only: severity 0=none, 1=warning, 2=fatal.
 * sdk_domain: 0=other, 1=core, 2=library, 3=resource, 4=stream.
 * No SDK error/debug text, URI, or credential is returned. */
typedef struct SeeonMediaDiagnostic {
  uint32_t severity, code, sdk_domain;
  int32_t sdk_code;
} SeeonMediaDiagnostic;

typedef struct SeeonMediaSourceStatus {
  SeeonMediaBinding binding;
  uint64_t frames, overwritten, dropped, malformed, tensor_absent, objects;
  uint32_t video_linked;
  SeeonMediaRecordTicket active_record;
} SeeonMediaSourceStatus;

typedef struct SeeonMediaStatus {
  SeeonMediaState state;
  SeeonMediaDiagnostic fatal, warning;
  uint64_t warnings, capacity_refusals, preview_dropped, late_record_callbacks;
  /* records_reserved counts occupied slots, including unread or quarantined results. */
  uint32_t source_count, callbacks_active, stop_timed_out, records_reserved;
  SeeonMediaSourceStatus sources[SEEON_MEDIA_MAX_SOURCES];
} SeeonMediaStatus;

/* The shared graph exclusively owns GLib's process-default context while its
 * control thread runs, including Smart Record duration timers. A competing
 * dispatcher/graph causes a startup STATE failure; there is no timer fallback.
 * One serialized caller owns all control/poll/destroy calls. SDK callbacks never
 * call Rust. OK on start/request means admitted, not PLAYING or sealed: poll
 * status/results. Request IDs are nonzero and strictly increasing per feature.
 * try_read_* copies into caller storage. TOO_SMALL never consumes a result;
 * preview fills its descriptor with the required size when only JPEG is small.
 * A delivered preview/record returns OK even when its .result describes failure.
 * Ready record completions remain drainable after a sticky pipeline failure.
 * Preview failure consumption does not release an unresolved encoder operation;
 * new requests return BUSY until its completion or a proven processing-bin drain.
 * Preview timeout_ms is 1..60000. Recording lookback must be less than the
 * configured cache; forward must be nonzero. Its deadline is forward+30 seconds
 * from admission. Session ID UINT32_MAX is reserved for failed start actions.
 */
SeeonMediaResult seeon_media_open(const SeeonMediaConfig *, SeeonMedia **,
                                 SeeonMediaDiagnostic *);
SeeonMediaResult seeon_media_start(SeeonMedia *);
SeeonMediaResult seeon_media_try_read_pose(SeeonMedia *, uint32_t source_id,
                                          SeeonMediaPose *, size_t);
SeeonMediaResult seeon_media_request_preview(SeeonMedia *, uint32_t source_id,
    const SeeonMediaBinding *, uint64_t request_id, uint32_t draw_objects,
    uint32_t timeout_ms);
SeeonMediaResult seeon_media_try_read_preview(SeeonMedia *, SeeonMediaPreview *,
    size_t descriptor_bytes, uint8_t *jpeg, size_t jpeg_capacity);
SeeonMediaResult seeon_media_record_start(SeeonMedia *, uint32_t source_id,
    const SeeonMediaBinding *, uint64_t request_id, uint32_t lookback_seconds,
    uint32_t forward_seconds, SeeonMediaRecordTicket *, size_t);
SeeonMediaResult seeon_media_record_stop(SeeonMedia *, uint32_t source_id,
    const SeeonMediaBinding *, uint64_t request_id, uint32_t session_id);
SeeonMediaResult seeon_media_try_read_record(SeeonMedia *, SeeonMediaRecord *, size_t);
SeeonMediaResult seeon_media_read_status(SeeonMedia *, SeeonMediaStatus *, size_t);
/* A relative monotonic deadline. Timeout retains the owner, native objects and
 * module; call stop again to reap them. This DSO is linked for the Worker process
 * lifetime, not an unloadable plugin: callback counters cannot prove DSO unload
 * safety. Never free unjoined objects. Stop cancels unfinished requests.
 * stop-sr is not completion, and even an sr-done filepath proves neither fsync
 * nor evidence sealing. */
SeeonMediaResult seeon_media_stop(SeeonMedia *, uint32_t deadline_ms);
SeeonMediaResult seeon_media_destroy(SeeonMedia *); /* Refuses until proven stopped. */

#ifdef __cplusplus
}
#endif
#endif
