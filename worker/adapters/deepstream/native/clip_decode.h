#ifndef SEEON_CLIP_DECODE_H
#define SEEON_CLIP_DECODE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Owns one demuxer, one single-threaded video decoder and one RGB24 scaler.
 * Frames keep their decoded size; conversion uses SWS_BILINEAR with the
 * scaler's default colourspace and range, as PyAV `to_ndarray("rgb24")` does.
 * No libav pointers cross this ABI.
 */
struct SeeonClipDecoder;

/* Return 0 on success; all errors are static, path-free text. `identity`
 * receives "libav-<LIBAVCODEC_IDENT>/<codec name>".
 */
int seeon_clipdec_open(const char *path, struct SeeonClipDecoder **decoder,
                       int32_t *width, int32_t *height, char *identity, size_t identity_size,
                       char *error, size_t error_size);
/* Return 1 with one packed RGB24 frame, 0 after the last frame, -1 on error.
 * `rgb` must hold width*height*3 bytes of the frame's size. `has_pts` is 0
 * when the decoder produced no timestamp. After -1 the decoder is unusable.
 */
int seeon_clipdec_next_rgb24(struct SeeonClipDecoder *decoder, uint8_t *rgb, size_t capacity,
                             int32_t *width, int32_t *height, int64_t *pts, int32_t *has_pts,
                             char *error, size_t error_size);
void seeon_clipdec_close(struct SeeonClipDecoder *decoder);

#ifdef __cplusplus
}
#endif
#endif
