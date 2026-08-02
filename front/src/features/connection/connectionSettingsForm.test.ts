import { describe, expect, it } from 'vitest';
import {
  buildConnectionPayload,
  connectionFormFromView,
  updateConnectionToken,
  updateFacilityId,
  validateConnectionForm,
  type ConnectionFormState,
} from '@/features/connection/connectionSettingsForm';
import type { ConnectionView } from '@/shared/api/types';

const blankForm: ConnectionFormState = {
  facility_id: '',
  facility_token: '',
};

const view: ConnectionView = {
  events_url: 'https://backend.example.com/v1/events',
  config_url: 'https://backend.example.com/v1/config',
  facility_id: 'facility-42',
  facility_token_set: true,
  facility_token_masked: '****ab12',
  configured: true,
  reachable: true,
  last_ok_at: '2026-08-01T00:00:00Z',
  updated_at: '2026-08-01T00:00:00Z',
};

describe('connectionFormFromView', () => {
  it('populates only the site-local facility ID from a saved view and never pre-fills the token', () => {
    expect(connectionFormFromView(view)).toEqual({
      facility_id: 'facility-42',
      facility_token: '',
    });
  });

  it('produces a blank form for a null view', () => {
    expect(connectionFormFromView(null)).toEqual(blankForm);
  });
});

describe('updateFacilityId', () => {
  it('updates the facility_id field', () => {
    const updated = updateFacilityId(blankForm, 'facility-7');
    expect(updated).toEqual({ ...blankForm, facility_id: 'facility-7' });
  });
});

describe('updateConnectionToken', () => {
  it('replaces the draft token value', () => {
    const updated = updateConnectionToken(blankForm, 'new-token');
    expect(updated).toEqual({ ...blankForm, facility_token: 'new-token' });
  });
});

describe('validateConnectionForm', () => {
  it('rejects a blank facility ID', () => {
    expect(validateConnectionForm(blankForm)).toBe('시설 ID를 입력하세요.');
  });

  it('rejects a whitespace-only facility ID', () => {
    expect(validateConnectionForm({ ...blankForm, facility_id: '   ' })).toBe('시설 ID를 입력하세요.');
  });

  it('accepts a non-empty facility ID', () => {
    expect(validateConnectionForm({ ...blankForm, facility_id: 'facility-42' })).toBeNull();
  });
});

describe('buildConnectionPayload', () => {
  it('sends the trimmed facility ID and omits an untouched token', () => {
    const form: ConnectionFormState = { facility_id: ' facility-42 ', facility_token: '' };
    const payload = buildConnectionPayload(form);

    expect(payload).toEqual({ facility_id: 'facility-42' });
    expect('facility_token' in payload).toBe(false);
  });

  it('never sends events_url or config_url -- they are packaging-time constants, not form fields', () => {
    const form = connectionFormFromView(view);
    const payload = buildConnectionPayload({ ...form, facility_id: 'facility-42' });

    expect('events_url' in payload).toBe(false);
    expect('config_url' in payload).toBe(false);
  });

  it('includes a typed token, trimmed', () => {
    const form = updateConnectionToken({ ...blankForm, facility_id: 'facility-42' }, '  new-secret  ');
    expect(buildConnectionPayload(form)).toEqual({ facility_id: 'facility-42', facility_token: 'new-secret' });
  });

  it('omits facility_token entirely when the technician never touched it', () => {
    const payload = buildConnectionPayload({ ...blankForm, facility_id: 'facility-42' });
    expect('facility_token' in payload).toBe(false);
  });
});
