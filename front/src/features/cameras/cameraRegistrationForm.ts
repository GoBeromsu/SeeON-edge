export type CameraRegistrationRtspForm = {
  scheme: 'rtsp' | 'rtsps';
  host: string;
  port: string;
  path: string;
  username: string;
  password: string;
  query: string;
};

export const initialCameraRegistrationRtspForm: CameraRegistrationRtspForm = {
  scheme: 'rtsp',
  host: '',
  port: '554',
  path: '/trackID=1',
  username: '',
  password: '',
  query: '',
};

function normalizePath(path: string): string {
  const trimmed = path.trim();
  if (!trimmed) return '/trackID=1';
  return trimmed.startsWith('/') ? trimmed : `/${trimmed}`;
}

function normalizeQuery(query: string): string {
  const trimmed = query.trim();
  if (!trimmed) return '';
  return trimmed.startsWith('?') ? trimmed.slice(1) : trimmed;
}

export function buildCameraRegistrationRtspUrl(form: CameraRegistrationRtspForm): string {
  const username = form.username.trim();
  const password = form.password;
  const auth = username ? `${encodeURIComponent(username)}${password ? `:${encodeURIComponent(password)}` : ''}@` : '';
  const port = form.port.trim();
  const query = normalizeQuery(form.query);
  return `${form.scheme}://${auth}${form.host.trim()}${port ? `:${port}` : ''}${normalizePath(form.path)}${query ? `?${query}` : ''}`;
}

function maskRtspQuery(rtspUrl: string): string {
  const fragmentStart = rtspUrl.indexOf('#');
  const urlWithoutFragment = fragmentStart < 0 ? rtspUrl : rtspUrl.slice(0, fragmentStart);
  const maskedFragment = fragmentStart < 0 ? '' : '#***';
  const queryStart = urlWithoutFragment.indexOf('?');
  if (queryStart < 0) return `${urlWithoutFragment}${maskedFragment}`;

  const maskedQuery = urlWithoutFragment
    .slice(queryStart + 1)
    .split('&')
    .map((part) => {
      const equalsIndex = part.indexOf('=');
      if (equalsIndex < 0) return '***';
      return `${part.slice(0, equalsIndex)}=***`;
    })
    .join('&');

  return `${urlWithoutFragment.slice(0, queryStart + 1)}${maskedQuery}${maskedFragment}`;
}

function maskRtspUserinfo(rtspUrl: string): string {
  const schemeEnd = rtspUrl.indexOf('://');
  if (schemeEnd < 0) return rtspUrl;

  const authorityStart = schemeEnd + 3;
  const authorityEndCandidate = rtspUrl.slice(authorityStart).search(/[/?#]/);
  const authorityEnd = authorityEndCandidate < 0 ? rtspUrl.length : authorityStart + authorityEndCandidate;
  const authority = rtspUrl.slice(authorityStart, authorityEnd);
  const userinfoEnd = authority.lastIndexOf('@');
  if (userinfoEnd < 0) return rtspUrl;

  return `${rtspUrl.slice(0, authorityStart)}***@${authority.slice(userinfoEnd + 1)}${rtspUrl.slice(authorityEnd)}`;
}

export function maskCameraRegistrationRtspPreview(rtspUrl: string): string {
  return maskRtspQuery(maskRtspUserinfo(rtspUrl));
}

// Mirrors the structure buildCameraRegistrationRtspUrl produces: scheme://[user[:pass]@]host[:port][path][?query]
const RTSP_URL_PATTERN = /^(rtsp|rtsps):\/\/(?:([^@/]*)@)?([^:/?#]+)(?::(\d*))?(\/[^?]*)?(?:\?(.*))?$/;

/**
 * Decomposes a full RTSP URL (e.g. pasted from vendor docs) into the 7 structured fields.
 * Returns null when the input isn't a recognizable rtsp(s):// URL — callers should leave the
 * individual fields untouched (and expanded) in that case rather than discarding the paste.
 * `parseCameraRegistrationRtspUrl(buildCameraRegistrationRtspUrl(form))` round-trips to `form`
 * for any form accepted by validateCameraRegistrationForm.
 */
export function parseCameraRegistrationRtspUrl(input: string): CameraRegistrationRtspForm | null {
  const trimmed = input.trim();
  if (!trimmed) return null;

  const match = RTSP_URL_PATTERN.exec(trimmed);
  if (!match) return null;

  const [, scheme, userinfo, host, port, path, query] = match;
  if (!host) return null;

  let username = '';
  let password = '';
  if (userinfo) {
    const colonIndex = userinfo.indexOf(':');
    try {
      if (colonIndex < 0) {
        username = decodeURIComponent(userinfo);
      } else {
        username = decodeURIComponent(userinfo.slice(0, colonIndex));
        password = decodeURIComponent(userinfo.slice(colonIndex + 1));
      }
    } catch {
      return null;
    }
  }

  return {
    scheme: scheme === 'rtsps' ? 'rtsps' : 'rtsp',
    host,
    port: port || '554',
    path: path || '/trackID=1',
    username,
    password,
    query: query ?? '',
  };
}

export function validateCameraRegistrationForm(label: string, rtspForm: CameraRegistrationRtspForm): string | null {
  if (!label.trim()) return '카메라 이름을 입력하세요.';
  if (!rtspForm.host.trim()) return '카메라 IP 또는 호스트를 입력하세요.';

  const rawPort = rtspForm.port;
  const port = Number(rawPort);
  if (!/^\d+$/.test(rawPort) || !Number.isInteger(port) || port < 1 || port > 65535) {
    return 'RTSP 포트는 1부터 65535 사이의 숫자여야 합니다.';
  }
  return null;
}

/**
 * Per spec R2.2, a label match is a warning, never a block (two cameras in the same room can
 * legitimately share a label). Only `rtsp_url` identity is checked server-side as a hard 409.
 * Comparison is case-insensitive and trims both sides so "301호 " and "301호" collide.
 */
export function findDuplicateLabelCamera<T extends { label: string }>(
  label: string,
  cameras: readonly T[],
): T | undefined {
  const normalized = label.trim().toLowerCase();
  if (!normalized) return undefined;
  return cameras.find((camera) => camera.label.trim().toLowerCase() === normalized);
}
