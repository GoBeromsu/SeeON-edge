# ADR 0001: Preserve the Source Stream for Evidence Clips

- Status: Accepted
- Date: 2026-07-16
- Scope: ML edge stream intake and event evidence clips

## Context

The incoming camera stream is the closest available representation of what the camera produced.
Changing its frame cadence, timestamps, resolution, codec, or compression can introduce timing drift,
quality loss, new failure modes, and additional CPU/GPU and memory pressure. Those effects are especially
costly on resource-constrained edge hosts and make evidence harder to compare with the source.

The ML pipeline must still decode frames for inference. That required decode is separate from the evidence
clip storage decision: decoding for analysis does not require the stored clip to be rebuilt from decoded
frames.

## Decision

The compressed input stream and its timestamps are authoritative for event evidence.

1. The primary evidence clip path MUST copy or remux source packets without transcoding, frame-rate
   conversion, frame dropping, resizing, or timestamp synthesis.
2. Source PTS/DTS and time base determine playback duration. A configured nominal FPS MUST NOT replace
   source timing or be used to compensate for missing frames.
3. Inference MAY decode the stream into frames, but decoded inference frames are not the primary evidence
   representation.
4. Keyframe boundaries MAY make a copied clip slightly longer than the requested event window. Prefer an
   expanded keyframe-aligned clip over silent transcoding.
5. A transformed derivative is allowed only when required for an explicit consumer constraint such as
   browser codec compatibility, redaction, or an overlay. It MUST be generated separately, identified as a
   derivative, and MUST NOT replace the preserved source clip.
6. Playback speed MUST remain 1x. The client MUST NOT hide timestamp or FPS defects with `playbackRate`.

"Preserve the source stream" does not mean avoiding all decode work. It means that analysis may decode the
stream while evidence storage retains the source packet cadence and encoding.

## Consequences

### Pros

- Avoids encoder-induced timing drift, frame duplication, frame loss, and generation loss.
- Removes evidence clip-encoding CPU/GPU work and reduces memory-copy pressure.
- Eliminates encoder configuration as a source of playback side effects.
- Retains the source codec, cadence, resolution, and timestamps for audit and comparison.
- Keeps the primary evidence path smaller and easier to reason about.

### Cons

- Browser or downstream codec/container support is not guaranteed; a derivative may still be required.
- Event boundaries are limited by the source GOP and keyframes, so exact cuts are not always possible.
- Variable frame rate, timestamp discontinuities, packet loss, and camera defects remain visible rather than
  being normalized away.
- Source bitrate directly determines storage and transfer cost.
- Redaction, overlays, resizing, and image enhancement require a separate transformed artifact.
- ML inference still incurs decode cost; this decision removes evidence re-encoding, not inference decode.

## Implementation Status

The worker now demuxes source video and optional audio packets into bounded camera-local rings while decoded
frames continue to feed analysis and snapshots. Primary clean clips select one reconnect epoch and codec
configuration at a decodable keyframe, then stream-copy the original packet payloads into MP4. Missing
history/future packets are recorded as truncation; missing keyframes, discontinuities, mixed configurations,
or remux verification failures fail closed without re-encoding.
