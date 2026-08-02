import { describe, expect, it } from 'vitest';
import {
  buildConnectionPayload,
  clearConnectionToken,
  connectionFormFromView,
  isAbsoluteHttpUrl,
  updateConnectionField,
  updateConnectionToken,
  validateConnectionForm,
  type ConnectionFormState,
} from '@/features/connection/connectionSettingsForm';
import type { ConnectionView } from '@/shared/api/types';

const blankForm: ConnectionFormState = {
  events_url: '',
  config_url: '',
  facility_id: '',
  facility_token: '',
  facility_token_cleared: false,
};

const view: ConnectionView = {
  events_url: 'https://backend.example.com/events',
  config_url: 'https://backend.example.com/config',
  facility_id: 'facility-42',
  facility_token_set: true,
  facility_token_masked: '****ab12',
  configured: true,
  reachable: true,
  last_ok_at: '2026-08-01T00:00:00Z',
  updated_at: '2026-08-01T00:00:00Z',
};

describe('isAbsoluteHttpUrl', () => {
  it.each([
    ['http://backend.local/config', true],
    ['https://backend.example.com/config', true],
    ['ftp://backend.local/config', false],
    ['backend.local/config', false],
    ['', false],
    ['   ', false],
    ['http://', false],
  ])('given %s, then returns %s', (value, expected) => {
    expect(isAbsoluteHttpUrl(value)).toBe(expected);
  });
});

describe('connectionFormFromView', () => {
  it('populates the WYSIWYG fields from a saved view and never pre-fills the token', () => {
    expect(connectionFormFromView(view)).toEqual({
      events_url: 'https://backend.example.com/events',
      config_url: 'https://backend.example.com/config',
      facility_id: 'facility-42',
      facility_token: '',
      facility_token_cleared: false,
    });
  });

  it('produces a blank form for a null view', () => {
    expect(connectionFormFromView(null)).toEqual(blankForm);
  });
});

describe('updateConnectionToken and clearConnectionToken', () => {
  it('typing a replacement token cancels a pending clear', () => {
    const cleared = clearConnectionToken(blankForm);
    expect(cleared.facility_token_cleared).toBe(true);

    const retyped = updateConnectionToken(cleared, 'new-token');
    expect(retyped).toEqual({ ...blankForm, facility_token: 'new-token', facility_token_cleared: false });
  });

  it('marks the token cleared and empties the draft value', () => {
    const typed = updateConnectionToken(blankForm, 'secret');
    expect(clearConnectionToken(typed)).toEqual({ ...blankForm, facility_token: '', facility_token_cleared: true });
  });
});

describe('validateConnectionForm', () => {
  it.each([
    [{ ...blankForm }, null],
    [{ ...blankForm, events_url: 'https://backend.example.com/events' }, null],
    [{ ...blankForm, events_url: 'not-a-url' }, '이벤트 URL 형식이 올바르지 않습니다. http:// 또는 https://로 시작하는 주소를 입력하세요.'],
    [{ ...blankForm, config_url: 'not-a-url' }, '설정 조회 URL 형식이 올바르지 않습니다. http:// 또는 https://로 시작하는 주소를 입력하세요.'],
    [{ ...blankForm, config_url: 'https://backend.example.com/config' }, null],
  ])('given %j, then returns %s', (form, expected) => {
    expect(validateConnectionForm(form)).toBe(expected);
  });
});

describe('updateConnectionField', () => {
  it.each((['events_url', 'config_url', 'facility_id'] as const))('updates only the %s field', (field) => {
    const updated = updateConnectionField(blankForm, field, 'new-value');
    expect(updated[field]).toBe('new-value');
    expect({ ...updated, [field]: '' }).toEqual(blankForm);
  });
});

describe('buildConnectionPayload', () => {
  it('sends every WYSIWYG field, trimmed, and omits an untouched token', () => {
    const form: ConnectionFormState = {
      events_url: '  https://backend.example.com/events  ',
      config_url: 'https://backend.example.com/config',
      facility_id: ' facility-42 ',
      facility_token: '',
      facility_token_cleared: false,
    };

    expect(buildConnectionPayload(form)).toEqual({
      events_url: 'https://backend.example.com/events',
      config_url: 'https://backend.example.com/config',
      facility_id: 'facility-42',
    });
  });

  it('clears WYSIWYG fields left empty back to null', () => {
    expect(buildConnectionPayload(blankForm)).toEqual({
      events_url: null,
      config_url: null,
      facility_id: null,
    });
  });

  it('includes a typed token', () => {
    const form = updateConnectionToken(blankForm, 'new-secret');
    expect(buildConnectionPayload(form)).toMatchObject({ facility_token: 'new-secret' });
  });

  it('includes an explicit null for a cleared token, even if cleared after typing', () => {
    const form = clearConnectionToken(updateConnectionToken(blankForm, 'discarded'));
    expect(buildConnectionPayload(form)).toMatchObject({ facility_token: null });
  });

  it('omits facility_token entirely when the technician never touched it', () => {
    const payload = buildConnectionPayload(blankForm);
    expect('facility_token' in payload).toBe(false);
  });
});
