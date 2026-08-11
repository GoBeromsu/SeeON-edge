import type { ConnectionInput, ConnectionView } from '@/shared/api/types';

const FACILITY_CODE = /^NH-[0-9A-HJKMNP-TV-Z]{10}$/;

export type ConnectionFormState = {
  readonly facility_code: string;
  readonly facility_token: string;
  readonly client_installation_ref: string;
};

export function connectionFormFromView(view: ConnectionView | null, generatedRef: string): ConnectionFormState {
  return {
    facility_code: view?.facility_code ?? '',
    facility_token: '',
    client_installation_ref: view?.client_installation_ref ?? generatedRef,
  };
}

export function updateFacilityCode(form: ConnectionFormState, value: string): ConnectionFormState {
  return { ...form, facility_code: value.toUpperCase() };
}

export function updateConnectionToken(form: ConnectionFormState, value: string): ConnectionFormState {
  return { ...form, facility_token: value };
}

export function validateConnectionForm(form: ConnectionFormState): string | null {
  if (!FACILITY_CODE.test(form.facility_code.trim())) return '시설 코드는 NH-로 시작하는 13자리 코드입니다.';
  if (!form.facility_token) return '시설 토큰을 입력하세요.';
  return null;
}

export function buildConnectionPayload(form: ConnectionFormState): ConnectionInput {
  return {
    facility_code: form.facility_code.trim(),
    facility_token: form.facility_token,
    client_installation_ref: form.client_installation_ref,
  };
}
