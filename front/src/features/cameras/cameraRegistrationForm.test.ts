import { describe, expect, it } from 'vitest';
import {
  buildCameraRegistrationRtspUrl,
  findDuplicateLabelCamera,
  initialCameraRegistrationRtspForm,
  parseCameraRegistrationRtspUrl,
  validateCameraRegistrationForm,
  type CameraRegistrationRtspForm,
} from '@/features/cameras/cameraRegistrationForm';

function form(overrides: Partial<CameraRegistrationRtspForm> = {}): CameraRegistrationRtspForm {
  return { ...initialCameraRegistrationRtspForm, ...overrides };
}

describe('validateCameraRegistrationForm', () => {
  it('requires a label', () => {
    expect(validateCameraRegistrationForm('', form({ host: 'camera.local' }))).toBe('카메라 이름을 입력하세요.');
  });

  it('requires a host', () => {
    expect(validateCameraRegistrationForm('301호', form())).toBe('카메라 IP 또는 호스트를 입력하세요.');
  });

  it('requires an in-range numeric port', () => {
    expect(validateCameraRegistrationForm('301호', form({ host: 'camera.local', port: '70000' }))).toContain('1부터 65535');
  });

  it('accepts a fully valid form', () => {
    expect(validateCameraRegistrationForm('301호', form({ host: 'camera.local' }))).toBeNull();
  });
});

describe('findDuplicateLabelCamera', () => {
  const roster = [
    { id: 'cam-1', label: '301호 침대 A' },
    { id: 'cam-2', label: '302호 침대 B' },
  ];

  it('finds an exact label match', () => {
    expect(findDuplicateLabelCamera('301호 침대 A', roster)).toBe(roster[0]);
  });

  it('matches case-insensitively', () => {
    const asciiRoster = [{ id: 'cam-1', label: 'Porch Camera' }];
    expect(findDuplicateLabelCamera('porch camera', asciiRoster)).toBe(asciiRoster[0]);
    expect(findDuplicateLabelCamera('PORCH CAMERA', asciiRoster)).toBe(asciiRoster[0]);
  });

  it('trims surrounding whitespace on both the input and the stored label before comparing', () => {
    const paddedRoster = [{ id: 'cam-1', label: '  301호 침대 A  ' }];
    expect(findDuplicateLabelCamera(' 301호 침대 A ', paddedRoster)).toBe(paddedRoster[0]);
  });

  it('returns undefined when no label matches', () => {
    expect(findDuplicateLabelCamera('303호 침대 C', roster)).toBeUndefined();
  });

  it('returns undefined for a blank or whitespace-only label instead of matching everything', () => {
    expect(findDuplicateLabelCamera('', roster)).toBeUndefined();
    expect(findDuplicateLabelCamera('   ', roster)).toBeUndefined();
  });

  it('returns undefined against an empty roster', () => {
    expect(findDuplicateLabelCamera('301호 침대 A', [])).toBeUndefined();
  });
});

describe('parseCameraRegistrationRtspUrl', () => {
  it('decomposes a full RTSP URL with credentials, a custom port, path, and query', () => {
    expect(parseCameraRegistrationRtspUrl('rtsp://operator:secret@192.0.2.10:8554/Streaming/Channels/101?subtype=0')).toEqual({
      scheme: 'rtsp',
      host: '192.0.2.10',
      port: '8554',
      path: '/Streaming/Channels/101',
      username: 'operator',
      password: 'secret',
      query: 'subtype=0',
    });
  });

  it('decomposes an rtsps URL with no credentials and no query', () => {
    expect(parseCameraRegistrationRtspUrl('rtsps://camera.local:554/trackID=1')).toEqual({
      scheme: 'rtsps',
      host: 'camera.local',
      port: '554',
      path: '/trackID=1',
      username: '',
      password: '',
      query: '',
    });
  });

  it('decodes percent-encoded credentials', () => {
    expect(parseCameraRegistrationRtspUrl('rtsp://operator%20name:p%40ss%2Fword@camera.local:554/trackID=1')).toMatchObject({
      username: 'operator name',
      password: 'p@ss/word',
    });
  });

  it('defaults a missing port and path', () => {
    expect(parseCameraRegistrationRtspUrl('rtsp://camera.local')).toEqual({
      scheme: 'rtsp',
      host: 'camera.local',
      port: '554',
      path: '/trackID=1',
      username: '',
      password: '',
      query: '',
    });
  });

  it('trims surrounding whitespace from a pasted URL', () => {
    expect(parseCameraRegistrationRtspUrl('  rtsp://camera.local:554/trackID=1  ')).not.toBeNull();
  });

  it.each([
    ['empty input', ''],
    ['whitespace only', '   '],
    ['a non-RTSP scheme', 'https://camera.local/stream'],
    ['free text', 'not a url at all'],
    ['a scheme with no host', 'rtsp://'],
  ])('returns null for %s', (_case, input) => {
    expect(parseCameraRegistrationRtspUrl(input)).toBeNull();
  });

  it('round-trips build -> parse back to the original form for a full-featured registration', () => {
    const original = form({
      scheme: 'rtsps',
      host: '192.0.2.10',
      port: '8554',
      path: '/Streaming/Channels/101',
      username: 'operator',
      password: 'secret',
      query: 'subtype=0',
    });

    expect(parseCameraRegistrationRtspUrl(buildCameraRegistrationRtspUrl(original))).toEqual(original);
  });

  it('round-trips build -> parse back to the original form for the bare default form', () => {
    const original = form({ host: 'camera.local' });

    expect(parseCameraRegistrationRtspUrl(buildCameraRegistrationRtspUrl(original))).toEqual(original);
  });

  it('round-trips build -> parse for credentials containing reserved characters', () => {
    const original = form({
      host: 'camera.local',
      username: 'operator name',
      password: 'p@ss/word:colon',
    });

    expect(parseCameraRegistrationRtspUrl(buildCameraRegistrationRtspUrl(original))).toEqual(original);
  });

  it('round-trips build -> parse for a username with no password', () => {
    const original = form({ host: 'camera.local', username: 'operator' });

    expect(parseCameraRegistrationRtspUrl(buildCameraRegistrationRtspUrl(original))).toEqual(original);
  });
});
