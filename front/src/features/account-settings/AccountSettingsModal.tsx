import { FormEvent, useRef, useState } from 'react';
import { updateDashboardCredentials } from '@/shared/api/client';
import { AccessibleDialog } from '@/shared/ui/AccessibleDialog';
import { useAuthSession } from '@/shared/ui/AuthGate';

type AccountSettingsModalProps = {
  open: boolean;
  onClose: () => void;
};

/** Single admin account, already authenticated — no current-password confirmation required. */
export function AccountSettingsModal({ open, onClose }: AccountSettingsModalProps): JSX.Element | null {
  const session = useAuthSession();
  const [newUsername, setNewUsername] = useState(session?.username ?? '');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [message, setMessage] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const usernameRef = useRef<HTMLInputElement>(null);

  function resetAndClose(): void {
    setNewUsername(session?.username ?? '');
    setNewPassword('');
    setConfirmPassword('');
    setMessage(null);
    setSuccess(false);
    setBusy(false);
    busyRef.current = false;
    onClose();
  }

  function requestClose(): void {
    if (busyRef.current) return;
    resetAndClose();
  }

  async function handleSubmit(event: FormEvent<HTMLFormElement>): Promise<void> {
    event.preventDefault();
    if (busyRef.current) return;
    setMessage(null);
    setSuccess(false);

    if (newPassword.length < 4) {
      setMessage('새 비밀번호는 4자 이상이어야 합니다.');
      return;
    }
    if (newPassword !== confirmPassword) {
      setMessage('새 비밀번호가 일치하지 않습니다.');
      return;
    }

    const trimmedUsername = newUsername.trim();
    busyRef.current = true;
    setBusy(true);
    try {
      await updateDashboardCredentials({
        username: trimmedUsername || undefined,
        newPassword,
      });
      session?.setUsername(trimmedUsername || session.username || '');
      setNewPassword('');
      setConfirmPassword('');
      setSuccess(true);
    } catch {
      setMessage('계정 정보를 변경하지 못했습니다. 잠시 후 다시 시도해 주세요.');
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }

  return (
    <AccessibleDialog open={open} title="계정 설정" onClose={requestClose} initialFocusRef={usernameRef}>
      <form onSubmit={(event) => void handleSubmit(event)} noValidate>
        <label>
          아이디
          <input
            ref={usernameRef}
            name="newUsername"
            autoComplete="username"
            value={newUsername}
            disabled={busy}
            onChange={(event) => setNewUsername(event.target.value)}
          />
        </label>

        <label>
          새 비밀번호
          <input
            name="newPassword"
            type="password"
            autoComplete="new-password"
            value={newPassword}
            disabled={busy}
            onChange={(event) => setNewPassword(event.target.value)}
          />
        </label>

        <label>
          새 비밀번호 확인
          <input
            name="confirmPassword"
            type="password"
            autoComplete="new-password"
            value={confirmPassword}
            disabled={busy}
            onChange={(event) => setConfirmPassword(event.target.value)}
          />
        </label>

        {message ? (
          <p className="auth-error" role="alert">
            {message}
          </p>
        ) : null}

        {success ? (
          <p className="dialog-success" role="status">
            계정 정보가 변경되었습니다.
          </p>
        ) : null}

        <div className="dialog-actions">
          <button type="button" className="dialog-secondary-action" disabled={busy} onClick={requestClose}>
            취소
          </button>
          <button type="submit" className="brand-action rounded-lg px-6 py-2 text-sm font-bold" disabled={busy}>
            {busy ? '변경 중...' : '변경하기'}
          </button>
        </div>
      </form>
    </AccessibleDialog>
  );
}
