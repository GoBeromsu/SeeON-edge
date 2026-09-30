//! Python `base64.b64decode` for the queue's `*_b64` fields, restricted to
//! padded standard base64 (RFC 4648 section 4).

/// Padded standard base64 (RFC 4648 section 4); anything else is refused.
pub(super) fn decode_base64(text: &str) -> Option<Vec<u8>> {
    let bytes = text.as_bytes();
    if !bytes.len().is_multiple_of(4) {
        return None;
    }
    let quads = bytes.len() / 4;
    let mut decoded = Vec::with_capacity(quads * 3);
    for (index, quad) in bytes.chunks(4).enumerate() {
        let pad = quad.iter().rev().take_while(|byte| **byte == b'=').count();
        if pad > 2 || (pad > 0 && index + 1 != quads) {
            return None;
        }
        let mut bits: u32 = 0;
        for byte in &quad[..4 - pad] {
            bits = (bits << 6) | u32::from(sextet(*byte)?);
        }
        bits <<= 6 * (pad as u32);
        decoded.extend_from_slice(&bits.to_be_bytes()[1..4 - pad]);
    }
    Some(decoded)
}

fn sextet(byte: u8) -> Option<u8> {
    match byte {
        b'A'..=b'Z' => Some(byte - b'A'),
        b'a'..=b'z' => Some(byte - b'a' + 26),
        b'0'..=b'9' => Some(byte - b'0' + 52),
        b'+' => Some(62),
        b'/' => Some(63),
        _ => None,
    }
}
