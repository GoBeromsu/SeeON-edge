import { describe, expect, it } from 'vitest';
import {
  buildConnectionPayload,
  connectionFormFromView,
  updateConnectionToken,
  updateFacilityCode,
  validateConnectionForm,
  type ConnectionFormState,
} from '@/features/connection/connectionSettingsForm';
import type { ConnectionView } from '@/shared/api/types';

const generatedRef = '78c1eaad-bd4b-44fc-88a1-401135d65a70';
const blank: ConnectionFormState = { facility_code: '', facility_token: '', client_installation_ref: generatedRef };
const view: ConnectionView = {
  events_url: 'https://backend.example.com/v1/events', config_url: 'https://backend.example.com/v1/config',
  facility_code: 'NH-7H2K9M4QXP', client_installation_ref: 'aa83ea3f-6e5f-4f45-a401-fb36c38835b6',
  facility_id: 'facility-42', edge_installation_id: 'c72bd9a7-3e04-47ba-a8cd-a56e54f98152',
  enrollment_generation: 3, facility_token_set: true, facility_token_masked: '****ab12',
  enrolled: true, configured: true, reachable: true, last_ok_at: null, updated_at: null,
};

describe('connection enrollment form', () => {
  it('loads the persisted code/reference while always leaving the token empty', () => {
    expect(connectionFormFromView(view, generatedRef)).toEqual({
      facility_code: 'NH-7H2K9M4QXP', facility_token: '', client_installation_ref: view.client_installation_ref,
    });
  });

  it('uses one generated reference before the server has persisted enrollment', () => {
    expect(connectionFormFromView(null, generatedRef)).toEqual(blank);
  });

  it('normalizes code casing without changing the opaque token', () => {
    const withCode = updateFacilityCode(blank, 'nh-7h2k9m4qxp');
    expect(updateConnectionToken(withCode, '  secret  ')).toEqual({
      ...blank, facility_code: 'NH-7H2K9M4QXP', facility_token: '  secret  ',
    });
  });

  it('requires the frozen facility-code shape and a token', () => {
    expect(validateConnectionForm(blank)).toContain('시설 코드');
    expect(validateConnectionForm({ ...blank, facility_code: 'facility-42', facility_token: 'secret' })).toContain('시설 코드');
    expect(validateConnectionForm({ ...blank, facility_code: 'NH-7H2K9M4QXP' })).toContain('토큰');
    expect(validateConnectionForm({ ...blank, facility_code: 'NH-7H2K9M4QXP', facility_token: 'secret' })).toBeNull();
  });

  it('builds exactly code, token, and installation reference without trimming the token', () => {
    expect(buildConnectionPayload({ ...blank, facility_code: ' NH-7H2K9M4QXP ', facility_token: ' secret ' })).toEqual({
      facility_code: 'NH-7H2K9M4QXP', facility_token: ' secret ', client_installation_ref: generatedRef,
    });
  });
});
