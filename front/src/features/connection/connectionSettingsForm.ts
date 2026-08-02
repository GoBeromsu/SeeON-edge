import type { ConnectionInput, ConnectionView } from '@/shared/api/types';

/**
 * Asymmetric field model, matching how each field can (and can't) be shown back to the user:
 * events_url/config_url/facility_id are WYSIWYG plain strings straight from the last-known
 * ConnectionView, so an empty box unambiguously means "clear this on save". facility_token is
 * different -- its true value is masked/never returned by the backend, so an empty box is
 * ambiguous between "left untouched" and "clear it"; `facility_token_cleared` disambiguates via
 * the explicit "지우기" action instead.
 */
export type ConnectionFormState = {
  events_url: string;
  config_url: string;
  facility_id: string;
  facility_token: string;
  facility_token_cleared: boolean;
};

export function connectionFormFromView(view: ConnectionView | null): ConnectionFormState {
  return {
    events_url: view?.events_url ?? '',
    config_url: view?.config_url ?? '',
    facility_id: view?.facility_id ?? '',
    facility_token: '',
    facility_token_cleared: false,
  };
}

export function updateConnectionField(
  form: ConnectionFormState,
  field: 'events_url' | 'config_url' | 'facility_id',
  value: string,
): ConnectionFormState {
  return { ...form, [field]: value };
}

/** Typing a new token implicitly cancels a pending "지우기" (the technician is replacing it, not clearing it). */
export function updateConnectionToken(form: ConnectionFormState, value: string): ConnectionFormState {
  return { ...form, facility_token: value, facility_token_cleared: false };
}

export function clearConnectionToken(form: ConnectionFormState): ConnectionFormState {
  return { ...form, facility_token: '', facility_token_cleared: true };
}

// Mirrors the backend's _is_http_url (backend/app/features/connection/router.py): absolute URL,
// scheme in {http, https}, non-empty netloc/host.
export function isAbsoluteHttpUrl(value: string): boolean {
  try {
    const url = new URL(value);
    return (url.protocol === 'http:' || url.protocol === 'https:') && url.host !== '';
  } catch {
    return false;
  }
}

export function validateConnectionForm(form: ConnectionFormState): string | null {
  if (form.events_url.trim() && !isAbsoluteHttpUrl(form.events_url.trim())) {
    return '이벤트 URL 형식이 올바르지 않습니다. http:// 또는 https://로 시작하는 주소를 입력하세요.';
  }
  if (form.config_url.trim() && !isAbsoluteHttpUrl(form.config_url.trim())) {
    return '설정 조회 URL 형식이 올바르지 않습니다. http:// 또는 https://로 시작하는 주소를 입력하세요.';
  }
  return null;
}

/**
 * events_url/config_url/facility_id are always included -- their trimmed text (or explicit null
 * when the box is empty, which the backend treats as "reset to the env-seed default"). Call only
 * after validateConnectionForm() returns null: this function does not itself reject malformed URLs.
 * facility_token is included only when the technician typed a replacement or explicitly cleared
 * it; otherwise it is omitted so the backend leaves the saved token untouched.
 */
export function buildConnectionPayload(form: ConnectionFormState): ConnectionInput {
  const payload: ConnectionInput = {
    events_url: form.events_url.trim() || null,
    config_url: form.config_url.trim() || null,
    facility_id: form.facility_id.trim() || null,
  };
  if (form.facility_token_cleared) {
    payload.facility_token = null;
  } else if (form.facility_token.trim()) {
    payload.facility_token = form.facility_token.trim();
  }
  return payload;
}
