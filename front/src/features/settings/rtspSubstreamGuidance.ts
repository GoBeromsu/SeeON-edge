const TRACK_ID_PATTERN = /trackid=(\d+)/i;

export type RtspSubstreamGuidance = {
  level: 'main-stream' | 'no-track-id';
  message: string;
};

/**
 * Client-side heuristic that nudges an operator toward the ML substream at
 * registration/edit time (issue #154). Deliberately advisory only, never
 * blocking: RTSP path conventions are vendor-specific and this registry has
 * no camera-vendor field to key off of, so a hard rejection would misfire on
 * every non-IDIS camera (and on any URL shape this heuristic fails to
 * recognize) while doing nothing about rows already registered with a
 * main-stream URL -- migrating those is a separate operational decision (see
 * issue #154).
 *
 * IDIS cameras expose two tracks on the same RTSP endpoint via a `trackID`
 * path segment: `trackID=1` is the full-resolution main stream, `trackID=2`
 * is the ML substream (H.265 640x360 30fps) -- see
 * .claude/skills/edge-bringup/references/worker-roster.md §3. A URL with no
 * `trackID` at all is flagged too, with softer wording: on an IDIS camera
 * that omits it the device defaults to the main stream, and for any other
 * vendor this is the only place a generic "use the low-resolution stream"
 * reminder can attach.
 */
export function assessRtspSubstreamGuidance(rtspUrl: string): RtspSubstreamGuidance | null {
  const trimmed = rtspUrl.trim();
  if (!trimmed) return null;
  const match = trimmed.match(TRACK_ID_PATTERN);
  if (!match) {
    return {
      level: 'no-track-id',
      message:
        'ML 파이프라인은 저해상도 서브스트림을 권장합니다. IDIS 카메라라면 주소 끝에 trackID=2를 추가하세요 (예: rtsp://.../trackID=2). 다른 제조사라면 보조/서브 스트림 경로를 확인하세요.',
    };
  }
  if (match[1] === '1') {
    return {
      level: 'main-stream',
      message:
        'trackID=1은 IDIS 카메라의 고해상도 메인 스트림입니다. ML 파이프라인은 저해상도 서브스트림을 사용합니다. 주소 끝을 trackID=2로 바꾸세요.',
    };
  }
  return null;
}
