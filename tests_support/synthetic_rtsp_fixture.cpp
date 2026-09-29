// Operator/test fixture only; never a production Worker or replay service.
// Mount the immutable synthetic corpus read-only at /corpus. The caller owns
// Docker-network isolation: expose no host ports. Each mount ends at file EOS.
#include <gst/gst.h>
#include <gst/rtsp-server/rtsp-server.h>
#include <glib-unix.h>

#include <array>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <initializer_list>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {
struct Fixture {
  const char *path;
  const char *mount;
  const char *launch;
};

constexpr std::array<Fixture, 4> kFixtures{{
    {"/corpus/a1-normal.mp4", "/a1-normal",
     "( filesrc location=/corpus/a1-normal.mp4 ! qtdemux name=demux "
     "demux.video_0 ! queue max-size-buffers=8 max-size-bytes=0 max-size-time=0 "
     "! h264parse ! rtph264pay name=pay0 pt=96 config-interval=-1 )"},
    {"/corpus/a1-fall.mp4", "/a1-fall",
     "( filesrc location=/corpus/a1-fall.mp4 ! qtdemux name=demux "
     "demux.video_0 ! queue max-size-buffers=8 max-size-bytes=0 max-size-time=0 "
     "! h264parse ! rtph264pay name=pay0 pt=96 config-interval=-1 )"},
    {"/corpus/a4-normal.mp4", "/a4-normal",
     "( filesrc location=/corpus/a4-normal.mp4 ! qtdemux name=demux "
     "demux.video_0 ! queue max-size-buffers=8 max-size-bytes=0 max-size-time=0 "
     "! h264parse ! rtph264pay name=pay0 pt=96 config-interval=-1 )"},
    {"/corpus/a4-fall.mp4", "/a4-fall",
     "( filesrc location=/corpus/a4-fall.mp4 ! qtdemux name=demux "
     "demux.video_0 ! queue max-size-buffers=8 max-size-bytes=0 max-size-time=0 "
     "! h264parse ! rtph264pay name=pay0 pt=96 config-interval=-1 )"},
}};

bool valid_file(const char *path) {
  // O_NONBLOCK also prevents an accidental FIFO from hanging startup.
  const int fd = ::open(path, O_RDONLY | O_CLOEXEC | O_NOFOLLOW | O_NONBLOCK);
  if (fd < 0) return false;
  struct stat info {};
  unsigned char header[12]{};
  const bool valid = ::fstat(fd, &info) == 0 && S_ISREG(info.st_mode) &&
      info.st_size >= 16 &&
      ::pread(fd, header, sizeof(header), 0) == static_cast<ssize_t>(sizeof(header)) &&
      std::memcmp(header + 4, "ftyp", 4) == 0;
  const bool closed = ::close(fd) == 0;
  // This is only input-file admission, not full MP4/GOP qualification. The real
  // qtdemux/h264parse pipeline validates the stream; the parent checks the GOP.
  return valid && closed;
}

bool required_plugins_present() {
  for (const char *name : {"filesrc", "qtdemux", "queue", "h264parse", "rtph264pay"}) {
    GstElementFactory *factory = gst_element_factory_find(name);
    if (!factory) return false;
    gst_object_unref(factory);
  }
  return true;
}

gboolean quit_loop(gpointer data) {
  g_main_loop_quit(static_cast<GMainLoop *>(data));
  return G_SOURCE_REMOVE;
}
} // namespace

int main() {
  struct stat corpus {};
  if (::lstat("/corpus", &corpus) != 0 || !S_ISDIR(corpus.st_mode)) {
    std::fputs("synthetic_rtsp_fixture startup_failed corpus_directory\n", stderr);
    return 1;
  }
  for (const auto &fixture : kFixtures) {
    if (!valid_file(fixture.path)) {
      std::fputs("synthetic_rtsp_fixture startup_failed corpus_file\n", stderr);
      return 1;
    }
  }

  GError *error = nullptr;
  const gboolean initialized = gst_init_check(nullptr, nullptr, &error);
  if (error) g_error_free(error);
  if (!initialized || !required_plugins_present()) {
    std::fputs("synthetic_rtsp_fixture startup_failed gstreamer\n", stderr);
    return 1;
  }

  GMainLoop *loop = g_main_loop_new(nullptr, FALSE);
  GstRTSPServer *server = gst_rtsp_server_new();
  gst_rtsp_server_set_address(server, "0.0.0.0");
  gst_rtsp_server_set_service(server, "8554");
  GstRTSPMountPoints *mounts = gst_rtsp_server_get_mount_points(server);
  for (const auto &fixture : kFixtures) {
    GstRTSPMediaFactory *factory = gst_rtsp_media_factory_new();
    gst_rtsp_media_factory_set_launch(factory, fixture.launch);
    gst_rtsp_media_factory_set_shared(factory, TRUE);
    gst_rtsp_media_factory_set_eos_shutdown(factory, TRUE);
    gst_rtsp_mount_points_add_factory(mounts, fixture.mount, factory);
  }
  g_object_unref(mounts);

  const guint attached = gst_rtsp_server_attach(server, nullptr);
  if (attached == 0) {
    std::fputs("synthetic_rtsp_fixture startup_failed attach\n", stderr);
    g_object_unref(server);
    g_main_loop_unref(loop);
    return 1;
  }

  const std::array<GSource *, 3> quit_sources{{
      g_timeout_source_new_seconds(90),
      g_unix_signal_source_new(SIGTERM),
      g_unix_signal_source_new(SIGINT),
  }};
  bool attached_quit_sources = true;
  for (GSource *source : quit_sources) {
    g_source_set_callback(source, quit_loop, loop, nullptr);
    if (g_source_attach(source, nullptr) == 0) attached_quit_sources = false;
  }
  if (attached_quit_sources) {
    std::puts("synthetic_rtsp_fixture started port=8554 mounts=4 lifetime_seconds=90");
    std::fflush(stdout);
    g_main_loop_run(loop);
  } else {
    std::fputs("synthetic_rtsp_fixture startup_failed shutdown_sources\n", stderr);
  }

  for (GSource *source : quit_sources) {
    g_source_destroy(source);
    g_source_unref(source);
  }
  g_source_remove(attached);
  g_object_unref(server);
  g_main_loop_unref(loop);
  std::puts("synthetic_rtsp_fixture exited");
  return attached_quit_sources ? 0 : 1;
}
