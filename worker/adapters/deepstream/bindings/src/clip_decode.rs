use std::ffi::{CString, c_char, c_int};
use std::marker::PhantomData;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr::NonNull;
use std::rc::Rc;

// Private ABI mirror of ../native/clip_decode.h.
mod ffi {
    use super::{c_char, c_int};

    #[repr(C)]
    pub struct Decoder {
        _opaque: [u8; 0],
    }
    unsafe extern "C" {
        pub fn seeon_clipdec_open(
            path: *const c_char,
            decoder: *mut *mut Decoder,
            width: *mut i32,
            height: *mut i32,
            identity: *mut c_char,
            identity_size: usize,
            error: *mut c_char,
            error_size: usize,
        ) -> c_int;
        pub fn seeon_clipdec_next_rgb24(
            decoder: *mut Decoder,
            rgb: *mut u8,
            capacity: usize,
            width: *mut i32,
            height: *mut i32,
            pts: *mut i64,
            has_pts: *mut i32,
            error: *mut c_char,
            error_size: usize,
        ) -> c_int;
        pub fn seeon_clipdec_close(decoder: *mut Decoder);
    }
}

/// Path-free clip decoding failures; there is no alternative decoder.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ClipError {
    InvalidPath,
    Unavailable,
    DecodeFailed,
    NativeContract,
    Poisoned,
}

impl std::fmt::Display for ClipError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(match self {
            Self::InvalidPath => "clip path must be nonempty and contain no NUL",
            Self::Unavailable => "clip is unreadable or its codec is unsupported",
            Self::DecodeFailed => "clip frame decode failed; decoder poisoned",
            Self::NativeContract => "native clip decoder result violated the ABI contract",
            Self::Poisoned => "clip decoder is unavailable after a failed native call",
        })
    }
}
impl std::error::Error for ClipError {}

/// One decoded frame as packed RGB24, row-major, `width * height * 3` bytes.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClipFrame {
    pub width: u32,
    pub height: u32,
    pub pts: Option<i64>,
    pub rgb: Vec<u8>,
}

/// Sole owner of one native demuxer, single-threaded decoder and SWS_BILINEAR
/// RGB24 scaler (the PyAV `to_ndarray("rgb24")` path). Not `Send` or `Sync`.
pub struct ClipDecoder {
    handle: NonNull<ffi::Decoder>,
    width: u32,
    height: u32,
    identity: String,
    poisoned: bool,
    _thread_confined: PhantomData<Rc<()>>,
}

impl ClipDecoder {
    /// Opens a regular local file through the libav file protocol only.
    pub fn open(path: &Path) -> Result<Self, ClipError> {
        let bytes = path.as_os_str().as_bytes();
        if bytes.is_empty() {
            return Err(ClipError::InvalidPath);
        }
        let path = CString::new(bytes).map_err(|_| ClipError::InvalidPath)?;
        let mut handle = std::ptr::null_mut();
        let (mut width, mut height) = (0_i32, 0_i32);
        let mut identity: [c_char; 128] = [0; 128];
        let mut error: [c_char; 256] = [0; 256];
        // SAFETY: the terminated path and every out-parameter live for this
        // synchronous call; native writes within the given sizes, owns partial
        // allocations on failure and returns a uniquely owned handle on success.
        let status = unsafe {
            ffi::seeon_clipdec_open(
                path.as_ptr(),
                &mut handle,
                &mut width,
                &mut height,
                identity.as_mut_ptr(),
                identity.len(),
                error.as_mut_ptr(),
                error.len(),
            )
        };
        if status != 0 {
            return Err(ClipError::Unavailable);
        }
        let handle = NonNull::new(handle).ok_or(ClipError::NativeContract)?;
        // From here Drop closes the handle, including on a contract failure below.
        let mut decoder = Self {
            handle,
            width: 0,
            height: 0,
            identity: String::new(),
            poisoned: false,
            _thread_confined: PhantomData,
        };
        decoder.identity = crate::fixed_text(&identity).map_err(|_| ClipError::NativeContract)?;
        (decoder.width, decoder.height) = frame_size(width, height)?;
        Ok(decoder)
    }

    pub fn width(&self) -> u32 {
        self.width
    }

    pub fn height(&self) -> u32 {
        self.height
    }

    /// `libav-<LIBAVCODEC_IDENT>/<codec name>`, recorded next to per-frame hashes.
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// The next frame in decode order, or `None` after the last one. Any
    /// native failure permanently poisons this decoder.
    pub fn next_frame(&mut self) -> Result<Option<ClipFrame>, ClipError> {
        if self.poisoned {
            return Err(ClipError::Poisoned);
        }
        let mut rgb = vec![0_u8; self.width as usize * self.height as usize * 3];
        let (mut width, mut height, mut pts, mut has_pts) = (0_i32, 0_i32, 0_i64, 0_i32);
        let mut error: [c_char; 256] = [0; 256];
        // SAFETY: exclusive self prevents overlapping calls or close. The buffer
        // holds exactly width*height*3 bytes as native requires, and every
        // out-parameter outlives this synchronous call.
        let status = unsafe {
            ffi::seeon_clipdec_next_rgb24(
                self.handle.as_ptr(),
                rgb.as_mut_ptr(),
                rgb.len(),
                &mut width,
                &mut height,
                &mut pts,
                &mut has_pts,
                error.as_mut_ptr(),
                error.len(),
            )
        };
        match status {
            0 => Ok(None),
            1 => {
                let frame = checked_frame(self.width, self.height, width, height, pts, has_pts);
                if frame.is_err() {
                    self.poisoned = true;
                }
                frame.map(|(width, height, pts)| {
                    Some(ClipFrame {
                        width,
                        height,
                        pts,
                        rgb,
                    })
                })
            }
            -1 => {
                self.poisoned = true;
                Err(ClipError::DecodeFailed)
            }
            _ => {
                self.poisoned = true;
                Err(ClipError::NativeContract)
            }
        }
    }
}

impl Drop for ClipDecoder {
    fn drop(&mut self) {
        // SAFETY: this non-cloneable owner closes exactly once on its opening
        // thread, and no call can overlap Drop.
        unsafe { ffi::seeon_clipdec_close(self.handle.as_ptr()) };
    }
}

fn frame_size(width: i32, height: i32) -> Result<(u32, u32), ClipError> {
    let width = u32::try_from(width).map_err(|_| ClipError::NativeContract)?;
    let height = u32::try_from(height).map_err(|_| ClipError::NativeContract)?;
    let bytes = (width as u64) * (height as u64) * 3;
    if width == 0 || height == 0 || bytes > 256 * 1024 * 1024 {
        return Err(ClipError::NativeContract);
    }
    Ok((width, height))
}

fn checked_frame(
    stream_width: u32,
    stream_height: u32,
    width: i32,
    height: i32,
    pts: i64,
    has_pts: i32,
) -> Result<(u32, u32, Option<i64>), ClipError> {
    if frame_size(width, height)? != (stream_width, stream_height) {
        return Err(ClipError::NativeContract);
    }
    let pts = match has_pts {
        0 => None,
        1 => Some(pts),
        _ => return Err(ClipError::NativeContract),
    };
    Ok((stream_width, stream_height, pts))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::ffi::OsStr;

    #[test]
    fn paths_are_rejected_locally_before_native_open() {
        for bytes in [b"".as_slice(), b"/clip\0.mp4".as_slice()] {
            assert_eq!(
                ClipDecoder::open(Path::new(OsStr::from_bytes(bytes))).err(),
                Some(ClipError::InvalidPath)
            );
        }
    }

    unsafe extern "C" {
        fn dlsym(handle: *mut std::ffi::c_void, symbol: *const c_char) -> *mut std::ffi::c_void;
    }

    // glibc's RTLD_DEFAULT is the null handle: the process global scope.
    fn global_symbol(name: &std::ffi::CStr) -> bool {
        // SAFETY: a terminated name and the documented RTLD_DEFAULT handle.
        !unsafe { dlsym(std::ptr::null_mut(), name.as_ptr()) }.is_null()
    }

    #[test]
    fn decoding_keeps_libav_and_its_libjpeg_out_of_the_global_scope() {
        // Two mid-grey 32x16 frames; the y4m demuxer and rawvideo decoder
        // ship in every libav build, so no media fixture is needed. swscale's
        // default limited-range BT.601 matrix, the PyAV to_ndarray path, maps
        // Y=U=V=128 to (128-16)*255/219 = 130.4, i.e. 130.
        let mut clip = b"YUV4MPEG2 W32 H16 F1:1 Ip A1:1 C420jpeg\n".to_vec();
        for _ in 0..2 {
            clip.extend_from_slice(b"FRAME\n");
            clip.extend(std::iter::repeat_n(128_u8, 32 * 16 + 2 * 16 * 8));
        }
        let path = std::env::temp_dir().join(format!("seeon-clipdec-{}.y4m", std::process::id()));
        std::fs::write(&path, &clip).expect("write the y4m clip");
        let decoded = (|| {
            let mut decoder = ClipDecoder::open(&path)?;
            let mut frames = Vec::new();
            while let Some(frame) = decoder.next_frame()? {
                frames.push(frame);
            }
            Ok::<_, ClipError>((decoder.identity().to_owned(), frames))
        })();
        std::fs::remove_file(&path).expect("remove the y4m clip");
        let (identity, frames) = decoded.expect("decode the y4m clip");
        assert!(
            identity.starts_with("libav-") && identity.ends_with("/rawvideo"),
            "{identity}"
        );
        assert_eq!(frames.len(), 2);
        for (index, frame) in frames.into_iter().enumerate() {
            assert_eq!(
                (frame.width, frame.height, frame.pts),
                (32, 16, Some(index as i64))
            );
            assert_eq!(frame.rgb.len(), 32 * 16 * 3);
            assert!(frame.rgb.iter().all(|&value| value == 130));
        }
        // DeepStream's nvjpegenc resolves jpeg_* through the global scope
        // first; system libjpeg-turbo there crashes it with an ABI mismatch.
        assert!(!global_symbol(c"avformat_open_input"));
        assert!(!global_symbol(c"jpeg_CreateCompress"));
    }

    #[test]
    fn native_frame_results_are_checked_against_the_stream() {
        assert_eq!(frame_size(640, 360), Ok((640, 360)));
        for (width, height) in [(0, 360), (640, 0), (-1, 360), (65536, 65536)] {
            assert_eq!(frame_size(width, height), Err(ClipError::NativeContract));
        }
        assert_eq!(
            checked_frame(640, 360, 640, 360, 7, 1),
            Ok((640, 360, Some(7)))
        );
        assert_eq!(
            checked_frame(640, 360, 640, 360, 7, 0),
            Ok((640, 360, None))
        );
        assert_eq!(
            checked_frame(640, 360, 320, 360, 7, 1),
            Err(ClipError::NativeContract)
        );
        assert_eq!(
            checked_frame(640, 360, 640, 360, 7, 2),
            Err(ClipError::NativeContract)
        );
    }
}
