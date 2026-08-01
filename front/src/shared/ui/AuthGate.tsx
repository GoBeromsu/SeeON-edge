import { createContext, FormEvent, ReactNode, useContext, useEffect, useState } from 'react';
import { fetchDashboardSession, loginDashboard, logoutDashboard } from '@/shared/api/client';
import { HttpError } from '@/shared/api/http';
import { SeniorAiLabBrand } from '@/shared/ui/SeniorAiLabBrand';

type AuthSession = {
  logout: () => Promise<void>;
};

const AuthSessionContext = createContext<AuthSession | null>(null);

export function useAuthSession(): AuthSession | null {
  return useContext(AuthSessionContext);
}

export function AuthGate({ children }: { children: ReactNode }): JSX.Element {
  const [loginId, setLoginId] = useState('');
  const [password, setPassword] = useState('');
  const [sessionState, setSessionState] = useState<'checking' | 'unavailable' | 'unauthorized' | 'authorized'>('checking');
  const [message, setMessage] = useState('');
  const [probeAttempt, setProbeAttempt] = useState(0);

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

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    setMessage('');
    if (!loginId.trim() || !password) {
      setMessage('아이디와 비밀번호를 입력해 주세요.');
      return;
    }
    try {
      await loginDashboard(loginId, password);
      setPassword('');
      setSessionState('authorized');
    } catch (error) {
      setMessage(
        error instanceof HttpError && [401, 403].includes(error.status)
          ? '아이디 또는 비밀번호가 올바르지 않습니다.'
          : '로그인 서비스에 연결하지 못했습니다. 잠시 후 다시 시도해 주세요.',
      );
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
    setSessionState('unauthorized');
    window.history.replaceState(null, '', `${window.location.pathname}?page=operations&wallPage=1${window.location.hash}`);
  }

  if (sessionState === 'authorized') {
    return (
      <AuthSessionContext.Provider value={{ logout }}>
        {children}
        {message ? <p className="auth-error" role="alert">{message}</p> : null}
      </AuthSessionContext.Provider>
    );
  }

  if (sessionState === 'checking') {
    return (
      <main className="auth-page" aria-busy="true">
        <section className="auth-card auth-config">
          <SeniorAiLabBrand />
          <p>로그인 상태를 확인하고 있습니다.</p>
        </section>
      </main>
    );
  }

  if (sessionState === 'unavailable') {
    return (
      <main className="auth-page">
        <section className="auth-card auth-config">
          <SeniorAiLabBrand />
          <h1>로그인 서비스 연결 실패</h1>
          <p>서버 상태를 확인한 뒤 다시 시도해 주세요.</p>
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
        </section>
      </main>
    );
  }

  return (
    <main className="auth-page">
      <section className="auth-card" aria-labelledby="login-title">
        <div className="auth-brand-panel">
          <SeniorAiLabBrand inverse />
          <p>카메라 운영과 안전 상태를 확인하는 관제 콘솔입니다.</p>
        </div>
        <form onSubmit={(event) => void handleSubmit(event)} noValidate>
          <h1 id="login-title">관리자 로그인</h1>
          <label>
            아이디
            <input name="loginId" required autoComplete="username" value={loginId} onChange={(event) => setLoginId(event.target.value)} />
          </label>
          <label>
            비밀번호
            <input name="password" type="password" required autoComplete="current-password" value={password} onChange={(event) => setPassword(event.target.value)} />
          </label>
          {message ? <p className="auth-error" role="alert">{message}</p> : null}
          <button type="submit" className="brand-action">로그인</button>
        </form>
      </section>
    </main>
  );
}
