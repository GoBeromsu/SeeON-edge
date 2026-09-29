#include "clip_decode.h"

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavcodec/version.h>
#include <libavformat/avformat.h>
#include <libavformat/version.h>
#include <libavutil/dict.h>
#include <libavutil/error.h>
#include <libavutil/frame.h>
#include <libavutil/macros.h>
#include <libavutil/version.h>
#include <libswscale/swscale.h>
#include <libswscale/version.h>
}
#include <dlfcn.h>
#include <sys/stat.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

namespace {
constexpr size_t kMaxFrameBytes = 256U * 1024U * 1024U;
void require(bool condition, const char *message) {
    if (!condition) throw std::runtime_error(message);
}
int error_text(char *buffer, size_t size, const char *message) noexcept {
    if (buffer && size) {
        const size_t length = std::min(size - 1, std::strlen(message));
        std::memcpy(buffer, message, length);
        buffer[length] = '\0';
    }
    return -1;
}

// libav is resolved at run time with RTLD_LOCAL instead of linked: a link-time
// dependency puts libav's closure, system libjpeg-turbo among it, in the
// process global scope, where DeepStream's nvjpegenc plugin then binds its
// jpeg_* calls to the wrong libjpeg ABI and crashes.
#define SEEON_LIBAV_FUNCTIONS(X) \
    X(avutil, av_dict_set) \
    X(avutil, av_dict_free) \
    X(avutil, av_frame_alloc) \
    X(avutil, av_frame_free) \
    X(avutil, av_frame_get_buffer) \
    X(avutil, av_frame_unref) \
    X(swscale, sws_getCachedContext) \
    X(swscale, sws_scale) \
    X(swscale, sws_freeContext) \
    X(avcodec, av_packet_alloc) \
    X(avcodec, av_packet_free) \
    X(avcodec, av_packet_unref) \
    X(avcodec, avcodec_find_decoder) \
    X(avcodec, avcodec_alloc_context3) \
    X(avcodec, avcodec_parameters_to_context) \
    X(avcodec, avcodec_open2) \
    X(avcodec, avcodec_send_packet) \
    X(avcodec, avcodec_receive_frame) \
    X(avcodec, avcodec_free_context) \
    X(avformat, avformat_open_input) \
    X(avformat, avformat_find_stream_info) \
    X(avformat, av_read_frame) \
    X(avformat, avformat_close_input)

struct Libav {
#define SEEON_LIBAV_SLOT(library, name) decltype(&::name) name = nullptr;
    SEEON_LIBAV_FUNCTIONS(SEEON_LIBAV_SLOT)
#undef SEEON_LIBAV_SLOT
};

// The sonames of the headers compiled against; the handles stay open for the
// process lifetime once every function resolved.
std::unique_ptr<const Libav> load_libav() {
    void *const handles[] = {
        dlopen("libavutil.so." AV_STRINGIFY(LIBAVUTIL_VERSION_MAJOR), RTLD_NOW | RTLD_LOCAL),
        dlopen("libswscale.so." AV_STRINGIFY(LIBSWSCALE_VERSION_MAJOR), RTLD_NOW | RTLD_LOCAL),
        dlopen("libavcodec.so." AV_STRINGIFY(LIBAVCODEC_VERSION_MAJOR), RTLD_NOW | RTLD_LOCAL),
        dlopen("libavformat.so." AV_STRINGIFY(LIBAVFORMAT_VERSION_MAJOR), RTLD_NOW | RTLD_LOCAL),
    };
    void *const avutil = handles[0];
    void *const swscale = handles[1];
    void *const avcodec = handles[2];
    void *const avformat = handles[3];
    auto library = std::make_unique<Libav>();
    bool resolved = avutil && swscale && avcodec && avformat;
#define SEEON_LIBAV_BIND(handle, name) \
    if (resolved) { \
        library->name = reinterpret_cast<decltype(&::name)>(dlsym(handle, #name)); \
        resolved = library->name != nullptr; \
    }
    SEEON_LIBAV_FUNCTIONS(SEEON_LIBAV_BIND)
#undef SEEON_LIBAV_BIND
    if (!resolved) {
        for (void *handle : handles) {
            if (handle) dlclose(handle);
        }
        return nullptr;
    }
    return library;
}

const Libav &libav() {
    static const std::unique_ptr<const Libav> library = load_libav();
    require(library != nullptr, "clip decoder runtime is unavailable");
    return *library;
}

struct PacketUnref {
    const Libav &av;
    AVPacket *packet;
    ~PacketUnref() { av.av_packet_unref(packet); }
};
struct FrameUnref {
    const Libav &av;
    AVFrame *frame;
    ~FrameUnref() { av.av_frame_unref(frame); }
};
} // namespace

struct SeeonClipDecoder {
    const Libav &av;
    AVFormatContext *format = nullptr;
    AVCodecContext *codec = nullptr;
    SwsContext *scaler = nullptr;
    AVPacket *packet = nullptr;
    AVFrame *frame = nullptr;
    AVFrame *rgb = nullptr;
    int stream = -1;
    int32_t width = 0;
    int32_t height = 0;
    bool draining = false;
    bool finished = false;
    bool failed = false;
    explicit SeeonClipDecoder(const Libav &library) : av(library) {}
    ~SeeonClipDecoder() {
        av.sws_freeContext(scaler);
        av.av_frame_free(&rgb);
        av.av_frame_free(&frame);
        av.av_packet_free(&packet);
        av.avcodec_free_context(&codec);
        av.avformat_close_input(&format);
    }
    // Packed RGB24 into the caller's buffer through an aligned scaler frame.
    void convert(uint8_t *out) {
        scaler = av.sws_getCachedContext(scaler, frame->width, frame->height, static_cast<AVPixelFormat>(frame->format),
                                         frame->width, frame->height, AV_PIX_FMT_RGB24, SWS_BILINEAR,
                                         nullptr, nullptr, nullptr);
        require(scaler != nullptr, "clip pixel format is unsupported");
        require(av.sws_scale(scaler, frame->data, frame->linesize, 0, frame->height, rgb->data, rgb->linesize) ==
                    frame->height, "clip colour conversion failed");
        const size_t row = static_cast<size_t>(width) * 3U;
        for (int32_t y = 0; y < height; ++y) {
            std::memcpy(out + static_cast<size_t>(y) * row, rgb->data[0] + static_cast<ptrdiff_t>(y) * rgb->linesize[0], row);
        }
    }
};

extern "C" int seeon_clipdec_open(const char *path, SeeonClipDecoder **out, int32_t *width, int32_t *height,
                                  char *identity, size_t identity_size, char *error, size_t error_size) {
    if (!out || !width || !height || !identity || !identity_size) {
        return error_text(error, error_size, "clip decoder outputs are required");
    }
    *out = nullptr;
    *width = 0;
    *height = 0;
    identity[0] = '\0';
    try {
        require(path && path[0], "clip path is required");
        struct stat info{};
        require(lstat(path, &info) == 0 && S_ISREG(info.st_mode) && info.st_size > 0, "clip is not a regular file");
        auto decoder = std::make_unique<SeeonClipDecoder>(libav());
        const Libav &av = decoder->av;
        // Only the file protocol; a path can never become a network source.
        const std::string location = std::string("file:") + path;
        AVDictionary *options = nullptr;
        av.av_dict_set(&options, "protocol_whitelist", "file", 0);
        const int opened = av.avformat_open_input(&decoder->format, location.c_str(), nullptr, &options);
        av.av_dict_free(&options);
        require(opened == 0, "clip container is not readable");
        require(av.avformat_find_stream_info(decoder->format, nullptr) >= 0, "clip stream info is unreadable");
        for (unsigned int i = 0; i < decoder->format->nb_streams; ++i) {
            if (decoder->format->streams[i]->codecpar->codec_type == AVMEDIA_TYPE_VIDEO) {
                decoder->stream = static_cast<int>(i);
                break;
            }
        }
        require(decoder->stream >= 0, "clip has no video stream");
        AVStream *stream = decoder->format->streams[decoder->stream];
        const AVCodec *codec = av.avcodec_find_decoder(stream->codecpar->codec_id);
        require(codec != nullptr, "clip codec is unsupported");
        decoder->codec = av.avcodec_alloc_context3(codec);
        require(decoder->codec != nullptr, "clip decoder allocation failed");
        require(av.avcodec_parameters_to_context(decoder->codec, stream->codecpar) >= 0, "clip decoder setup failed");
        decoder->codec->thread_count = 1;
        decoder->codec->thread_type = 0;
        decoder->codec->pkt_timebase = stream->time_base;
        require(av.avcodec_open2(decoder->codec, codec, nullptr) == 0, "clip decoder setup failed");
        decoder->width = stream->codecpar->width;
        decoder->height = stream->codecpar->height;
        require(decoder->width > 0 && decoder->height > 0 &&
                static_cast<uint64_t>(decoder->width) * static_cast<uint64_t>(decoder->height) * 3U <= kMaxFrameBytes,
                "clip frame size is invalid");
        decoder->packet = av.av_packet_alloc();
        decoder->frame = av.av_frame_alloc();
        decoder->rgb = av.av_frame_alloc();
        require(decoder->packet && decoder->frame && decoder->rgb, "clip decoder allocation failed");
        decoder->rgb->format = AV_PIX_FMT_RGB24;
        decoder->rgb->width = decoder->width;
        decoder->rgb->height = decoder->height;
        require(av.av_frame_get_buffer(decoder->rgb, 0) == 0, "clip decoder allocation failed");
        const std::string name = std::string("libav-" LIBAVCODEC_IDENT "/") + codec->name;
        require(name.size() < identity_size, "decoder identity buffer is too small");
        std::memcpy(identity, name.c_str(), name.size() + 1);
        *width = decoder->width;
        *height = decoder->height;
        *out = decoder.release();
        if (error && error_size) error[0] = '\0';
        return 0;
    } catch (const std::runtime_error &failure) {
        identity[0] = '\0';
        return error_text(error, error_size, failure.what());
    } catch (...) {
        identity[0] = '\0';
        return error_text(error, error_size, "clip decoder admission failed");
    }
}

extern "C" int seeon_clipdec_next_rgb24(SeeonClipDecoder *decoder, uint8_t *rgb, size_t capacity,
                                        int32_t *width, int32_t *height, int64_t *pts, int32_t *has_pts,
                                        char *error, size_t error_size) {
    if (!decoder) return error_text(error, error_size, "clip decoder is required");
    try {
        require(!decoder->failed, "clip decoder is unavailable after a failed frame");
        require(rgb && width && height && pts && has_pts, "clip frame outputs are required");
        require(capacity == static_cast<size_t>(decoder->width) * static_cast<size_t>(decoder->height) * 3U,
                "clip frame buffer size does not match the stream");
        if (decoder->finished) return 0;
        const Libav &av = decoder->av;
        // Same order as PyAV decode(stream): drain frames, feed one packet, flush at end.
        while (true) {
            const int received = av.avcodec_receive_frame(decoder->codec, decoder->frame);
            if (received == 0) {
                FrameUnref guard{av, decoder->frame};
                require(decoder->frame->width == decoder->width && decoder->frame->height == decoder->height,
                        "clip frame size changed mid-stream");
                decoder->convert(rgb);
                *width = decoder->width;
                *height = decoder->height;
                *has_pts = decoder->frame->pts == AV_NOPTS_VALUE ? 0 : 1;
                *pts = *has_pts ? decoder->frame->pts : 0;
                if (error && error_size) error[0] = '\0';
                return 1;
            }
            if (received == AVERROR_EOF) {
                decoder->finished = true;
                if (error && error_size) error[0] = '\0';
                return 0;
            }
            require(received == AVERROR(EAGAIN) && !decoder->draining, "clip frame decode failed");
            const int read = av.av_read_frame(decoder->format, decoder->packet);
            if (read == AVERROR_EOF) {
                decoder->draining = true;
                require(av.avcodec_send_packet(decoder->codec, nullptr) == 0, "clip decoder flush failed");
                continue;
            }
            require(read >= 0, "clip demux failed");
            PacketUnref guard{av, decoder->packet};
            if (decoder->packet->stream_index != decoder->stream) continue;
            require(av.avcodec_send_packet(decoder->codec, decoder->packet) == 0, "clip frame decode failed");
        }
    } catch (const std::runtime_error &failure) {
        decoder->failed = true;
        return error_text(error, error_size, failure.what());
    } catch (...) {
        decoder->failed = true;
        return error_text(error, error_size, "clip frame decode failed");
    }
}

extern "C" void seeon_clipdec_close(SeeonClipDecoder *decoder) { delete decoder; }
