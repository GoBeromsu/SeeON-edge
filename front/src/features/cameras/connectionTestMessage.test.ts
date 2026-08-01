import { describe, expect, it } from 'vitest';
import { connectionFailureMessage } from '@/features/cameras/connectionTestMessage';

describe('connectionFailureMessage', () => {
  it.each([
    ['timeout', '카메라 응답 시간이 초과되었습니다. 네트워크와 전원을 확인하세요.'],
    ['auth', '카메라 인증에 실패했습니다. 아이디와 비밀번호를 확인하세요.'],
    ['decode', '영상 형식을 처리할 수 없습니다. 카메라 스트림 설정을 확인하세요.'],
  ])('maps error_class %s to a specific Korean reason', (errorClass, expected) => {
    expect(connectionFailureMessage(errorClass)).toBe(expected);
  });

  it.each([undefined, '', 'unknown_class'])('falls back to a generic message for %j', (errorClass) => {
    expect(connectionFailureMessage(errorClass)).toBe('카메라 연결을 확인할 수 없습니다. 네트워크와 설정을 확인하세요.');
  });
});
