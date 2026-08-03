import { createContext, FormEvent, ReactNode, useContext, useEffect, useRef, useState } from 'react';
import { fetchDashboardSession, loginDashboard, logoutDashboard } from '@/shared/api/client';
import { HttpError, subscribeUnauthorized } from '@/shared/api/http';

const BRAND_TEXT = 'Senior AI Lab Edge';

type AuthSession = {
  logout: () => Promise<void>;
  /**
   * The username entered for the current login, if the user actually logged
   * in during this browser tab's session. Not carried across a page reload
   * (the session-check endpoint returns no body), so account-settings prefill
   * only works within the same tab lifetime that performed the login.
   */
  username: string | null;
  /** Updates the displayed username after a successful credential rotation. */
  setUsername: (username: string) => void;
};

const AuthSessionContext = createContext<AuthSession | null>(null);

export function useAuthSession(): AuthSession | null {
  return useContext(AuthSessionContext);
}

function AuthShell({ busy = false, children }: { busy?: boolean; children: ReactNode }): JSX.Element {
  return (
    <main className="auth-page" aria-busy={busy || undefined}>
      <section className="auth-card">
        <div className="auth-card-header">
          <h1 id="auth-title">{BRAND_TEXT}</h1>
        </div>
        <div className="auth-card-body">{children}</div>
      </section>
    </main>
  );
}

export function AuthGate({ children }: { children: ReactNode }): JSX.Element {
  const [loginId, setLoginId] = useState('');
  const [password, setPassword] = useState('');
  const [username, setUsername] = useState<string | null>(null);
  const [sessionState, setSessionState] = useState<'checking' | 'unavailable' | 'unauthorized' | 'authorized'>('checking');
  const [message, setMessage] = useState('');
  const [submitting, setSubmitting] = useState(false);
  const [probeAttempt, setProbeAttempt] = useState(0);
  const loginIdRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let active = true;
    void fetchDashboardSession()
      .then(() => {
        if (active) setSessionState('authorized');
      })
      .catch((error: unknown) => {
        if (!active) return;
        if (error instanceof HttpError && [401, 403].includes(error.status)) {
          setSessionState('unauthorized');
        } else {
          setSessionState('unavailable');
        }
      });
    return () => {
      active = false;
    };
  }, [probeAttempt]);

  useEffect(() => {
    if (sessionState !== 'unauthorized') return;
    loginIdRef.current?.focus();
  }, [sessionState]);

  /**
   * A 401/403 on any request (not just the mount-time probe) means the session died mid-flight —
   * e.g. a backend restart invalidating cookies while the operator is watching the wall. Flip
   * straight back to the login form instead of leaving a silently-frozen authorized view up.
   */
  useEffect(() => subscribeUnauthorized(() => {
    setSessionState((current) => (current === 'authorized' ? 'unauthorized' : current));
  }), []);

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setMessage('');
    if (!loginId.trim() || !password) {
      setMessage('아이디와 비밀번호를 입력해 주세요.');
      return;
    }
    setSubmitting(true);
    try {
      await loginDashboard(loginId, password);
      setUsername(loginId.trim());
      setPassword('');
      setSessionState('authorized');
    } catch (error) {
      setMessage(
        error instanceof HttpError && [401, 403].includes(error.status)
          ? '아이디 또는 비밀번호가 올바르지 않습니다.'
          : '로그인 서비스에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.',
      );
    } finally {
      setSubmitting(false);
    }
  }

  async function logout(): Promise<void> {
    setMessage('');
    try {
      await logoutDashboard();
    } catch {
      setMessage('로그아웃하지 못했습니다. 다시 시도해 주세요.');
      return;
    }
    setLoginId('');
    setPassword('');
    setUsername(null);
    setSessionState('unauthorized');
    window.history.replaceState(null, '', `${window.location.pathname}?page=operations${window.location.hash}`);
  }

  if (sessionState === 'authorized') {
    return (
      <AuthSessionContext.Provider value={{ logout, username, setUsername }}>
        {children}
        {message ? <p className="auth-error auth-error-shell" role="alert">{message}</p> : null}
      </AuthSessionContext.Provider>
    );
  }

  if (sessionState === 'checking') {
    return (
      <AuthShell busy>
        <p className="auth-status" role="status">로그인 상태를 확인하고 있습니다.</p>
      </AuthShell>
    );
  }

  if (sessionState === 'unavailable') {
    return (
      <AuthShell>
        <p className="text-sm font-semibold text-destructive" role="alert">로그인 서비스 연결 실패</p>
        <p className="auth-status">서버 상태를 확인한 뒤 다시 시도해 주세요.</p>
        <button
          type="button"
          className="brand-action"
          onClick={() => {
            setSessionState('checking');
            setProbeAttempt((attempt) => attempt + 1);
          }}
        >
          다시 시도
        </button>
      </AuthShell>
    );
  }

  return (
    <AuthShell>
      <form onSubmit={(event) => void handleSubmit(event)} noValidate aria-labelledby="auth-title">
        <label htmlFor="login-id">
          아이디
          <input
            ref={loginIdRef}
            id="login-id"
            name="loginId"
            required
            autoComplete="username"
            value={loginId}
            disabled={submitting}
            onChange={(event) => setLoginId(event.target.value)}
          />
        </label>
        <label htmlFor="login-password">
          비밀번호
          <input
            id="login-password"
            name="password"
            type="password"
            required
            autoComplete="current-password"
            value={password}
            disabled={submitting}
            onChange={(event) => setPassword(event.target.value)}
          />
        </label>
        <p className={`auth-error${message ? '' : ' auth-error-slot'}`} role={message ? 'alert' : undefined}>
          {message || '\u00a0'}
        </p>
        <button type="submit" className="brand-action" disabled={submitting}>
          {submitting ? '확인 중…' : '로그인'}
        </button>
      </form>
    </AuthShell>
  );
}
