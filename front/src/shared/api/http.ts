import { getApiBase } from '@/shared/api/session';

export class HttpError extends Error {
  constructor(readonly status: number, readonly body?: unknown) {
    super(`HTTP ${status}`);
  }
}

export async function requestJson(path: string, init?: RequestInit): Promise<unknown> {
  const response = await fetch(`${getApiBase()}${path}`, {
    credentials: 'same-origin',
    ...init,
    headers: {
      Accept: 'application/json',
      ...(init?.body ? { 'Content-Type': 'application/json' } : {}),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    let body: unknown;
    try {
      body = await response.json();
    } catch {
      body = undefined;
    }
    throw new HttpError(response.status, body);
  }

  if (response.status === 204) {
    return undefined;
  }

  return response.json();
}
