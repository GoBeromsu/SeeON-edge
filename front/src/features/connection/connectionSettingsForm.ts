import type { ConnectionInput, ConnectionView } from '@/shared/api/types';

/**
 * The backend address (events/config URLs) is a packaging-time constant baked into the edge's
 * Docker/compose env, not a per-site input -- so the only things a technician configures here are
 * the site-local values. Token semantics are 2-state, unlike the WYSIWYG facility_id: the token's
 * true value is masked/never returned by the backend, so an empty box means "leave the saved token
 * untouched" and a typed value means "replace it". There is no explicit clear affordance in this
 * form; the API still supports clearing a token via an explicit null, it's just not exposed here.
 */
export type ConnectionFormState = {
  facility_id: string;
  facility_token: string;
};

export function connectionFormFromView(view: ConnectionView | null): ConnectionFormState {
  return {
    facility_id: view?.facility_id ?? '',
    facility_token: '',
  };
}

export function updateFacilityId(form: ConnectionFormState, value: string): ConnectionFormState {
  return { ...form, facility_id: value };
}

export function updateConnectionToken(form: ConnectionFormState, value: string): ConnectionFormState {
  return { ...form, facility_token: value };
}

export function validateConnectionForm(form: ConnectionFormState): string | null {
  if (!form.facility_id.trim()) {
    return '시설 ID를 입력하세요.';
  }
  return null;
}

/**
 * events_url/config_url are packaging-time constants and are never sent from this form -- the
 * backend's partial-update semantics (an omitted key is left untouched) already keep them as-is.
 * facility_token is included only when the technician typed a replacement; otherwise it is omitted
 * so the backend leaves the saved token untouched. Call only after validateConnectionForm() returns
 * null: this function does not itself reject a blank facility_id.
 */
export function buildConnectionPayload(form: ConnectionFormState): ConnectionInput {
  const payload: ConnectionInput = {
    facility_id: form.facility_id.trim(),
  };
  if (form.facility_token.trim()) {
    payload.facility_token = form.facility_token.trim();
  }
  return payload;
}
