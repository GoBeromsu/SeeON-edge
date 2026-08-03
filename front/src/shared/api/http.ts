import { getApiBase } from '@/shared/api/session';

export class HttpError extends Error {
  constructor(readonly status: number, readonly body?: unknown) {
    super(`HTTP ${status}`);
  }
}

const unauthorizedListeners = new Set<() => void>();

/**
 * Global 401/403 signal so UI shells (AuthGate) can react to a session going invalid mid-flight
 * (e.g. backend restart invalidating cookies) instead of only probing once on mount. Every
 * `requestJson` call that fails with 401/403 notifies subscribers before throwing.
 */
export function subscribeUnauthorized(listener: () => void): () => void {
  unauthorizedListeners.add(listener);
  return () => unauthorizedListeners.delete(listener);
}

function notifyUnauthorized(): void {
  unauthorizedListeners.forEach((listener) => listener());
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
    if (response.status === 401 || response.status === 403) {
      notifyUnauthorized();
    }
    throw new HttpError(response.status, body);
  }

  if (response.status === 204) {
    return undefined;
  }

  return response.json();
}
