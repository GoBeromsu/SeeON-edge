//! Original G002 test transport, not native/runtime qualification.
//! No dependencies, SDK, UUID generator/normalizer, mailbox, or admission oracle.
//!
//! v1 binary grammar: integers are little endian; text is u32 byte length followed
//! by exact UTF-8 (including empty strings, NULs and newlines). Binding is text
//! boot, text normalized child UUID, text camera, u64 generation, u64 epoch,
//! text transform. Watermark is i64 PTS, u64 canonical seq, u64 native ordinal.
//! Request: `SRCADM01`, u32 operation_count (1..=128), text transport, binding
//! for initial registration, then operation_count-1 operations:
//!   1: binding (explicit re-registration)
//!   2: text transport, binding (frame identity), u8 PTS-present (0/1), optional
//!      i64 PTS, u64 canonical seq, u64 native ordinal.
//! Exact EOF and <=128 KiB input are required. Frame UUID normalization belongs
//! to the caller (the differential uses actual Python UUID/str before encoding).
//!
//! Response: same magic/count, then after EVERY operation: u8 result,
//! u8 owner-present; if present, text transport, binding, u8 actual is_ready,
//! u8 high-water-present and optional watermark; finally u8 retained-present and
//! optional binding/watermark from the most recent AcceptedWork. A successful
//! re-registration clears this test-held header; refusal retains it. No policy
//! lifetime, rotation/floor, or take/remove comparison is made.
//! Results: 0 registered, 1 accepted, 2 malformed camera, 3 invalid child UUID,
//! 4 unknown source, 5 boot, 6 child, 7 generation, 8 epoch, 9 transform,
//! 10 transport mismatch, 11 PTS missing, 12 discarded publication,
//! 13 non-increasing PTS, 14 canonical seq, 15 native ordinal,
//! 16 regressing fence, 17 generation exhausted, 18 epoch exhausted.
//! These are exact Rust results; only the Python comparison maps 13..=15 to its
//! observable `late` counter, without changing acceptance. Invalid registration
//! is a Rust-only control, not parity with Python's permissive dataclass.
//!
//! A failed initial registration may only be a one-operation request. The entire
//! response is buffered and capped at 1 MiB. Malformed/over-budget transport
//! exits 2 with empty stdout and exactly `source-admission-probe: rejected\n` on
//! stderr. Valid transport (including semantic refusals) exits 0, empty stderr.
//! The caller independently bounds subprocess input/output and wall time.
#![forbid(unsafe_code)]

use seeon_worker::source_admission::{
    AcceptedWork, AdmissionError, FrameMetadata, HighWater, SourceAdmission, SourceBinding,
};
use std::io::{Read, Write};

const MAGIC: &[u8; 8] = b"SRCADM01";
const MAX_INPUT: usize = 128 * 1024;
const MAX_OUTPUT: usize = 1024 * 1024;
const MAX_OPERATIONS: usize = 128;
type Result<T> = std::result::Result<T, ()>;

struct Cursor<'a> {
    input: &'a [u8],
    position: usize,
}

impl<'a> Cursor<'a> {
    fn take(&mut self, count: usize) -> Result<&'a [u8]> {
        let end = self.position.checked_add(count).ok_or(())?;
        let bytes = self.input.get(self.position..end).ok_or(())?;
        self.position = end;
        Ok(bytes)
    }

    fn bytes<const N: usize>(&mut self) -> Result<[u8; N]> {
        self.take(N)?.try_into().map_err(|_| ())
    }

    fn byte(&mut self) -> Result<u8> {
        Ok(self.bytes::<1>()?[0])
    }

    fn text(&mut self) -> Result<&'a str> {
        let count = u32::from_le_bytes(self.bytes()?) as usize;
        std::str::from_utf8(self.take(count)?).map_err(|_| ())
    }

    fn binding(&mut self) -> Result<SourceBinding> {
        Ok(SourceBinding {
            worker_boot_id: self.text()?.into(),
            child_instance_id: self.text()?.into(),
            camera_id: self.text()?.into(),
            source_generation: u64::from_le_bytes(self.bytes()?),
            stream_epoch: u64::from_le_bytes(self.bytes()?),
            transform_id: self.text()?.into(),
        })
    }
}

fn status(error: AdmissionError) -> u8 {
    match error {
        AdmissionError::MalformedCamera => 2,
        AdmissionError::InvalidChildInstanceId => 3,
        AdmissionError::UnknownSource => 4,
        AdmissionError::BootMismatch => 5,
        AdmissionError::ChildMismatch => 6,
        AdmissionError::GenerationMismatch => 7,
        AdmissionError::EpochMismatch => 8,
        AdmissionError::TransformMismatch => 9,
        AdmissionError::TransportMismatch => 10,
        AdmissionError::PtsMissing => 11,
        AdmissionError::DiscardedPublication => 12,
        AdmissionError::NonIncreasingPts => 13,
        AdmissionError::NonIncreasingCanonicalSequence => 14,
        AdmissionError::NonIncreasingNativePublicationSequence => 15,
        AdmissionError::RegressingFence => 16,
        AdmissionError::GenerationExhausted => 17,
        AdmissionError::EpochExhausted => 18,
    }
}

fn binding_size(binding: &SourceBinding) -> usize {
    32 + binding.worker_boot_id.len()
        + binding.child_instance_id.len()
        + binding.camera_id.len()
        + binding.transform_id.len()
}

fn put_text(out: &mut Vec<u8>, text: &str) {
    out.extend_from_slice(&(text.len() as u32).to_le_bytes());
    out.extend_from_slice(text.as_bytes());
}

fn put_binding(out: &mut Vec<u8>, binding: &SourceBinding) {
    put_text(out, &binding.worker_boot_id);
    put_text(out, &binding.child_instance_id);
    put_text(out, &binding.camera_id);
    out.extend_from_slice(&binding.source_generation.to_le_bytes());
    out.extend_from_slice(&binding.stream_epoch.to_le_bytes());
    put_text(out, &binding.transform_id);
}

fn put_water(out: &mut Vec<u8>, water: HighWater) {
    out.extend_from_slice(&water.source_pts.to_le_bytes());
    out.extend_from_slice(&water.canonical_sequence.to_le_bytes());
    out.extend_from_slice(&water.native_publish_sequence.to_le_bytes());
}

fn snapshot(
    out: &mut Vec<u8>,
    code: u8,
    owner: Option<&SourceAdmission>,
    retained: Option<&AcceptedWork>,
) -> Result<()> {
    // All strings originate in the bounded input, so these sums cannot overflow.
    let mut additional = 3;
    if let Some(owner) = owner {
        additional += 6 + owner.transport_id().len() + binding_size(owner.binding());
        if owner.high_water().is_some() {
            additional += 24;
        }
    }
    if let Some(work) = retained {
        additional += binding_size(work.binding()) + 24;
    }
    if additional > MAX_OUTPUT - out.len() {
        return Err(());
    }
    out.push(code);
    out.push(u8::from(owner.is_some()));
    if let Some(owner) = owner {
        put_text(out, owner.transport_id());
        put_binding(out, owner.binding());
        out.push(u8::from(owner.is_ready()));
        out.push(u8::from(owner.high_water().is_some()));
        if let Some(water) = owner.high_water() {
            put_water(out, water);
        }
    }
    out.push(u8::from(retained.is_some()));
    if let Some(work) = retained {
        put_binding(out, work.binding());
        put_water(out, work.high_water());
    }
    Ok(())
}

fn operation(
    cursor: &mut Cursor<'_>,
    owner: &mut SourceAdmission,
    retained: &mut Option<AcceptedWork>,
) -> Result<u8> {
    match cursor.byte()? {
        1 => match owner.reregister(cursor.binding()?) {
            Ok(()) => {
                *retained = None;
                Ok(0)
            }
            Err(error) => Ok(status(error)),
        },
        2 => {
            let transport_id = cursor.text()?;
            let binding = cursor.binding()?;
            let source_pts = match cursor.byte()? {
                0 => None,
                1 => Some(i64::from_le_bytes(cursor.bytes()?)),
                _ => return Err(()),
            };
            let frame = FrameMetadata {
                transport_id,
                worker_boot_id: &binding.worker_boot_id,
                child_instance_id: &binding.child_instance_id,
                camera_id: &binding.camera_id,
                source_generation: binding.source_generation,
                stream_epoch: binding.stream_epoch,
                transform_id: &binding.transform_id,
                source_pts,
                canonical_sequence: u64::from_le_bytes(cursor.bytes()?),
                native_publish_sequence: u64::from_le_bytes(cursor.bytes()?),
            };
            match owner.admit(frame) {
                Ok(work) => {
                    *retained = Some(work);
                    Ok(1)
                }
                Err(error) => Ok(status(error)),
            }
        }
        _ => Err(()),
    }
}

fn run(input: &[u8]) -> Result<Vec<u8>> {
    if input.len() > MAX_INPUT {
        return Err(());
    }
    let mut cursor = Cursor { input, position: 0 };
    if cursor.take(MAGIC.len())? != MAGIC {
        return Err(());
    }
    let count = u32::from_le_bytes(cursor.bytes()?) as usize;
    if count == 0 || count > MAX_OPERATIONS {
        return Err(());
    }
    let transport = cursor.text()?.to_owned();
    let (mut owner, code) = match SourceAdmission::new(transport, cursor.binding()?) {
        Ok(owner) => (Some(owner), 0),
        Err(error) => (None, status(error)),
    };
    let mut retained = None;
    let mut out = MAGIC.to_vec();
    out.extend_from_slice(&(count as u32).to_le_bytes());
    snapshot(&mut out, code, owner.as_ref(), retained.as_ref())?;
    for _ in 1..count {
        let code = operation(&mut cursor, owner.as_mut().ok_or(())?, &mut retained)?;
        snapshot(&mut out, code, owner.as_ref(), retained.as_ref())?;
    }
    if cursor.position != input.len() {
        return Err(());
    }
    Ok(out)
}

fn main() {
    let result = (|| {
        let mut input = Vec::new();
        std::io::stdin()
            .lock()
            .take((MAX_INPUT + 1) as u64)
            .read_to_end(&mut input)
            .map_err(|_| ())?;
        let output = run(&input)?;
        std::io::stdout().lock().write_all(&output).map_err(|_| ())
    })();
    if result.is_err() {
        let _ = std::io::stderr()
            .lock()
            .write_all(b"source-admission-probe: rejected\n");
        std::process::exit(2);
    }
}
