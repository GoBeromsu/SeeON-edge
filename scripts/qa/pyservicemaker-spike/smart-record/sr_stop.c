/* Smart Record stop semantics at the plugin level (nvurisrcbin action signals).
 *
 * Mirrors deepstream-testsr: start-sr(sessionId*, startTime, duration, user),
 * later stop-sr(sessionId), and "sr-done" delivering NvDsSRRecordingInfo.
 * Built to separate a pyservicemaker binding defect from a plugin behaviour.
 */
#include <gst/gst.h>
#include <glib.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "gst-nvdssr.h"

static GMainLoop *loop;
static GstElement *mux;
static gint64 t0;
static volatile gint linked = 0;

static gdouble now(void) { return (g_get_monotonic_time() - t0) / 1e6; }

static void on_pad(GstElement *src, GstPad *pad, gpointer d) {
  GstCaps *caps = gst_pad_get_current_caps(pad);
  if (!caps) caps = gst_pad_query_caps(pad, NULL);
  const gchar *name = gst_structure_get_name(gst_caps_get_structure(caps, 0));
  if (g_str_has_prefix(name, "video")) {
    GstPad *sink = gst_element_request_pad_simple(mux, "sink_0");
    g_print("[%7.2f] link=%d\n", now(), gst_pad_link(pad, sink));
    g_atomic_int_set(&linked, 1);
  }
  gst_caps_unref(caps);
}

static void on_sr_done(GstElement *src, NvDsSRRecordingInfo *info, gpointer user) {
  g_print("[%7.2f] sr-done session=%u file=%s dir=%s duration_ms=%lu w=%u\n", now(),
          info->sessionId, info->filename, info->dirpath, (unsigned long)info->duration, info->width);
}

static gboolean on_bus(GstBus *bus, GstMessage *msg, gpointer d) {
  if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_ERROR) {
    GError *e; gchar *dbg; gst_message_parse_error(msg, &e, &dbg);
    g_print("[%7.2f] ERROR %s\n", now(), e->message);
    g_main_loop_quit(loop);
  } else if (GST_MESSAGE_TYPE(msg) == GST_MESSAGE_EOS) {
    g_main_loop_quit(loop);
  }
  return TRUE;
}

static gpointer drive(gpointer src_) {
  GstElement *src = src_;
  const gchar *mode = getenv("SR_MODE") ? getenv("SR_MODE") : "stop";
  /* Gate on source liveness, not wall-clock: SR caches encoded frames, so a
   * start-sr before the first frame arrives records nothing. Then let the
   * cache fill past one I-frame. */
  for (int i = 0; i < 300 && !g_atomic_int_get(&linked); i++) g_usleep(100 * 1000);
  if (!g_atomic_int_get(&linked)) { g_print("[%7.2f] source never linked\n", now()); }
  g_usleep(10 * G_USEC_PER_SEC);
  NvDsSRSessionId sess = 0;
  if (strcmp(mode, "stop") == 0) {
    g_signal_emit_by_name(src, "start-sr", &sess, 0, 20, NULL);
    g_print("[%7.2f] start-sr(0,20) session=%u\n", now(), sess);
    g_usleep(4 * G_USEC_PER_SEC);
    g_signal_emit_by_name(src, "stop-sr", sess);
    g_print("[%7.2f] stop-sr(%u)\n", now(), sess);
    g_usleep(10 * G_USEC_PER_SEC);
  } else if (strcmp(mode, "overlap") == 0) {
    g_signal_emit_by_name(src, "start-sr", &sess, 5, 6, NULL);
    g_print("[%7.2f] start-sr(5,6) session=%u\n", now(), sess);
    g_usleep(3 * G_USEC_PER_SEC);
    NvDsSRSessionId sess2 = 0;
    g_signal_emit_by_name(src, "start-sr", &sess2, 0, 6, NULL);
    g_print("[%7.2f] start-sr(0,6) while inflight session=%u\n", now(), sess2);
    g_usleep(14 * G_USEC_PER_SEC);
  } else {
    g_signal_emit_by_name(src, "start-sr", &sess, 5, 6, NULL);
    g_print("[%7.2f] start-sr(5,6) session=%u\n", now(), sess);
    g_usleep(14 * G_USEC_PER_SEC);
  }
  g_print("[%7.2f] sending EOS\n", now());
  gst_element_send_event(GST_ELEMENT(gst_element_get_parent(src)), gst_event_new_eos());
  return NULL;
}

int main(int argc, char **argv) {
  gst_init(&argc, &argv);
  t0 = g_get_monotonic_time();
  const gchar *uri = getenv("SR_RTSP_URI");
  const gchar *out = getenv("SR_OUT_DIR") ? getenv("SR_OUT_DIR") : "/work/records";
  GstElement *pipe = gst_pipeline_new("sr-c");
  GstElement *src = gst_element_factory_make("nvurisrcbin", "src");
  mux = gst_element_factory_make("nvstreammux", "mux");
  GstElement *sink = gst_element_factory_make("fakesink", "sink");
  g_object_set(src, "uri", uri, "select-rtp-protocol", 4, "latency", 200,
               "smart-record", 2, "smart-rec-cache", 20, "init-rtsp-reconnect-interval", 5, "rtsp-reconnect-interval", 5, "smart-rec-dir-path", out,
               "smart-rec-container", 0, NULL);
  g_object_set(mux, "batch-size", 1, "width", 640, "height", 360, "live-source", TRUE,
               "batched-push-timeout", 40000, NULL);
  g_object_set(sink, "sync", FALSE, NULL);
  gst_bin_add_many(GST_BIN(pipe), src, mux, sink, NULL);
  gst_element_link(mux, sink);
  g_signal_connect(src, "pad-added", G_CALLBACK(on_pad), NULL);
  g_signal_connect(src, "sr-done", G_CALLBACK(on_sr_done), NULL);
  loop = g_main_loop_new(NULL, FALSE);
  gst_bus_add_watch(gst_pipeline_get_bus(GST_PIPELINE(pipe)), on_bus, NULL);
  g_thread_new("drive", drive, src);
  gst_element_set_state(pipe, GST_STATE_PLAYING);
  g_main_loop_run(loop);
  gst_element_set_state(pipe, GST_STATE_NULL);
  return 0;
}
