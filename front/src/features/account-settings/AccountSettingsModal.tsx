import { FormEvent, useRef, useState } from 'react';
import { updateDashboardCredentials } from '@/shared/api/client';
import { HttpError } from '@/shared/api/http';
import { AccessibleDialog } from '@/shared/ui/AccessibleDialog';
import { useAuthSession } from '@/shared/ui/AuthGate';

type AccountSettingsModalProps = {
  open: boolean;
  onClose: () => void;
};

export function AccountSettingsModal({ open, onClose }: AccountSettingsModalProps): JSX.Element | null {
  const session = useAuthSession();
  const [currentPassword, setCurrentPassword] = useState('');
  const [newUsername, setNewUsername] = useState(session?.username ?? '');
  const [newPassword, setNewPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [message, setMessage] = useState<string | null>(null);
  const [success, setSuccess] = useState(false);
  const [busy, setBusy] = useState(false);
  const busyRef = useRef(false);
  const currentPasswordRef = useRef<HTMLInputElement>(null);

  function resetAndClose(): void {
    setCurrentPassword('');
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

    if (!currentPassword) {
      setMessage('현재 비밀번호를 입력해 주세요.');
      return;
    }
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
        currentPassword,
        newUsername: trimmedUsername || undefined,
        newPassword,
      });
      session?.setUsername(trimmedUsername || session.username || '');
      setCurrentPassword('');
      setNewPassword('');
      setConfirmPassword('');
      setSuccess(true);
    } catch (error) {
      setMessage(
        error instanceof HttpError && error.status === 403
          ? '현재 비밀번호가 올바르지 않습니다.'
          : '계정 정보를 변경하지 못했습니다. 잠시 후 다시 시도해 주세요.',
      );
    } finally {
      busyRef.current = false;
      setBusy(false);
    }
  }

  return (
    <AccessibleDialog open={open} title="계정 설정" onClose={requestClose} initialFocusRef={currentPasswordRef}>
      <div className="flex items-start justify-between gap-4">
        <p className="text-sm font-bold text-brand">로그인 아이디/비밀번호 변경</p>
        <button type="button" disabled={busy} onClick={requestClose} className="rounded-full bg-surface2 px-4 py-2 text-sm font-bold text-ink-soft hover:bg-surface2 disabled:cursor-not-allowed disabled:opacity-60">
          닫기
        </button>
      </div>

      <form className="mt-6 space-y-4" onSubmit={(event) => void handleSubmit(event)} noValidate>
        <label className="block text-sm font-bold text-ink-soft">
          현재 비밀번호
          <input
            ref={currentPasswordRef}
            name="currentPassword"
            type="password"
            autoComplete="current-password"
            value={currentPassword}
            disabled={busy}
            onChange={(event) => setCurrentPassword(event.target.value)}
            className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4"
          />
        </label>

        <label className="block text-sm font-bold text-ink-soft">
          새 아이디 (선택)
          <input
            name="newUsername"
            autoComplete="username"
            value={newUsername}
            disabled={busy}
            onChange={(event) => setNewUsername(event.target.value)}
            className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4"
            placeholder="비워두면 현재 아이디를 유지합니다"
          />
        </label>

        <label className="block text-sm font-bold text-ink-soft">
          새 비밀번호
          <input
            name="newPassword"
            type="password"
            autoComplete="new-password"
            value={newPassword}
            disabled={busy}
            onChange={(event) => setNewPassword(event.target.value)}
            className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4"
          />
        </label>

        <label className="block text-sm font-bold text-ink-soft">
          새 비밀번호 확인
          <input
            name="confirmPassword"
            type="password"
            autoComplete="new-password"
            value={confirmPassword}
            disabled={busy}
            onChange={(event) => setConfirmPassword(event.target.value)}
            className="mt-2 w-full rounded-2xl border border-border bg-surface2 px-4 py-3 text-ink outline-none ring-brand focus:ring-4"
          />
        </label>

        <button type="submit" disabled={busy} className="brand-action rounded-lg px-6 py-3 text-sm font-black shadow-sm disabled:cursor-not-allowed disabled:opacity-60">
          {busy ? '변경 중...' : '변경하기'}
        </button>
      </form>

      {message ? (
        <p className="mt-5 rounded-2xl bg-status-dangerBg px-4 py-3 text-sm font-bold text-status-danger" role="alert">
          {message}
        </p>
      ) : null}

      {success ? (
        <p className="mt-5 rounded-2xl bg-brand-soft px-4 py-3 text-sm font-bold text-brand" role="status">
          계정 정보가 변경되었습니다.
        </p>
      ) : null}
    </AccessibleDialog>
  );
}
