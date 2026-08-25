import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { confirmTopologyPreview, fetchTopologyPreview, type TopologyPreview } from '@/shared/api/topologyClient';
import { HttpError } from '@/shared/api/http';
import { TopologyConfirmationDialog } from '@/features/cameras/TopologyConfirmationDialog';

vi.mock('@/shared/api/topologyClient', async () => {
  const actual = await vi.importActual<typeof import('@/shared/api/topologyClient')>('@/shared/api/topologyClient');
  return { ...actual, confirmTopologyPreview: vi.fn(), fetchTopologyPreview: vi.fn() };
});

const preview: TopologyPreview = {
  confirmation_id: '0197f671-3a31-7a6c-a6e4-83ed412de81b',
  digest: 'a'.repeat(64),
  expires_at: '2099-01-01T00:00:00.000Z',
  snapshot_id: '0197f671-3a31-7a6c-a6e4-83ed412de81c',
  client_revision: 4,
  server_revision: 9,
  cameras: 2,
  rooms: 1,
  floors: 0,
  confirmed: false,
};

function button(scope: ParentNode, label: string): HTMLButtonElement {
  const match = Array.from(scope.querySelectorAll('button')).find((candidate) => candidate.textContent?.trim() === label);
  if (!match) throw new Error(`missing button ${label}`);
  return match;
}

async function renderDialog(currentPreview: TopologyPreview | null, onConfirmed = vi.fn()): Promise<{ readonly host: HTMLElement; readonly root: Root; readonly onConfirmed: ReturnType<typeof vi.fn> }> {
  const container = document.createElement('div');
  document.body.append(container);
  const root = createRoot(container);
  await act(async () => {
    root.render(
      <TopologyConfirmationDialog open preview={currentPreview} onClose={() => undefined} onConfirmed={onConfirmed} onChanged={() => undefined} />,
    );
  });
  // AccessibleDialog renders its content via createPortal(..., document.body), so the actual
  // dialog markup (including the confirm button and any role="alert" text) is a sibling of
  // `container` directly under document.body -- not a descendant of it. Assertions must query
  // the real portaled dialog element, not the disconnected React-root container.
  const host = document.body.querySelector<HTMLElement>('[role="dialog"]');
  if (!host) throw new Error('dialog did not render into document.body');
  return { host, root, onConfirmed };
}

beforeEach(() => {
  vi.mocked(fetchTopologyPreview).mockResolvedValue({ preview });
  vi.mocked(confirmTopologyPreview).mockResolvedValue({ snapshot_id: preview.snapshot_id, client_revision: 4, server_revision: 10 });
});

afterEach(() => {
  document.body.innerHTML = '';
  vi.clearAllMocks();
});

describe('TopologyConfirmationDialog', () => {
  it('shows the deactivation counts but keeps the server revision behind a details disclosure', async () => {
    const { host } = await renderDialog(preview);

    expect(host.textContent).toContain('카메라');
    expect(host.textContent).toContain('2대');
    expect(host.textContent).toContain('방');
    expect(host.textContent).toContain('1개');
    const details = host.querySelector('details');
    expect(details?.textContent).toContain('서버 리비전');
    expect(details?.textContent).toContain('9');
  });

  it('rechecks the exact preview before one explicit confirmation and blocks double submit', async () => {
    let finish: (() => void) | undefined;
    vi.mocked(confirmTopologyPreview).mockReturnValue(new Promise((resolve) => {
      finish = () => resolve({ snapshot_id: preview.snapshot_id, client_revision: 4, server_revision: 10 });
    }));
    const { host } = await renderDialog(preview);

    const confirm = button(host, '제외 확정');
    // A synchronous second click lands while busyRef is already true (set before the first
    // `await fetchTopologyPreview()`), so the component blocks the resubmit before it ever
    // re-fetches the preview -- not merely before the confirm call. That's one recheck, not two.
    await act(async () => { confirm.click(); confirm.click(); await Promise.resolve(); await Promise.resolve(); });

    expect(fetchTopologyPreview).toHaveBeenCalledTimes(1);
    expect(confirmTopologyPreview).toHaveBeenCalledTimes(1);
    expect(confirmTopologyPreview).toHaveBeenCalledWith({
      confirmation_id: preview.confirmation_id,
      digest: preview.digest,
      client_revision: 4,
      server_revision: 9,
    });
    await act(async () => finish?.());
  });

  it('refuses confirmation when the durable preview changed after the dialog opened', async () => {
    const changed = { ...preview, server_revision: 10 };
    vi.mocked(fetchTopologyPreview).mockResolvedValueOnce({ preview: changed });
    const { host } = await renderDialog(preview);

    await act(async () => button(host, '제외 확정').click());

    expect(confirmTopologyPreview).not.toHaveBeenCalled();
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('변경되었습니다');
  });

  it.each([
    [401, '등록 인증'],
    [403, '권한'],
    [409, '바뀌었습니다'],
    [410, '기한이 지났습니다'],
    [503, '연결할 수 없습니다'],
  ] as const)('classifies a %i confirm failure distinctly', async (status, expectedFragment) => {
    vi.mocked(confirmTopologyPreview).mockRejectedValue(new HttpError(status, { detail: 'internal body' }));
    const { host } = await renderDialog(preview);

    await act(async () => button(host, '제외 확정').click());

    expect(host.querySelector('[role="alert"]')?.textContent).toContain(expectedFragment);
    expect(host.textContent).not.toContain('internal body');
  });

  it('disables confirmation once the preview has expired', async () => {
    const expired = { ...preview, expires_at: '2000-01-01T00:00:00.000Z' };
    const { host } = await renderDialog(expired);

    expect(button(host, '제외 확정').disabled).toBe(true);
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('확인 기한이 지났습니다');
  });
});
