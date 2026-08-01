import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { createCamera, deleteCamera, fetchCameras, testCamera, updateCamera, updateCameraDecodeBackend, type Camera } from '@/shared/api/client';
import { CameraManagementPage } from '@/features/camera-management/CameraManagementPage';

vi.mock('@/shared/api/client', async () => {
  const actual = await vi.importActual<typeof import('@/shared/api/client')>('@/shared/api/client');
  return {
    ...actual,
    createCamera: vi.fn(),
    deleteCamera: vi.fn(),
    fetchCameras: vi.fn(),
    testCamera: vi.fn(),
    updateCamera: vi.fn(),
    updateCameraDecodeBackend: vi.fn(),
  };
});

const camera: Camera = { id: 'cam-1', label: '301호 A', rtsp_url_masked: 'rtsp://user:***@redacted-camera/live', space_id: '301', space_name: '301호', floor_name: '3층', backend_camera_id: null, status: 'online', created_at: null };
const createdCamera = { ...camera, id: 'cam-2', label: '302호 A', space_id: '302', space_name: '302호' };

function setInput(name: string, value: string): void {
  const input = document.querySelector(`input[name="${name}"]`);
  if (!(input instanceof HTMLInputElement)) throw new Error(`missing input ${name}`);
  act(() => {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value')?.set?.call(input, value);
    input.dispatchEvent(new Event('input', { bubbles: true }));
  });
}

async function renderPage(): Promise<{ host: HTMLDivElement; root: ReturnType<typeof createRoot> }> {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  act(() => root.render(<CameraManagementPage />));
  await act(async () => {});
  return { host, root };
}

beforeEach(() => {
  vi.mocked(fetchCameras).mockReset();
  vi.mocked(deleteCamera).mockReset();
  vi.mocked(createCamera).mockReset();
  vi.mocked(testCamera).mockReset();
  vi.mocked(updateCamera).mockReset();
  vi.mocked(updateCameraDecodeBackend).mockReset();
});

afterEach(() => { document.body.innerHTML = ''; });

describe('CameraManagementPage', () => {
  it('shows first loading, successful empty, and no unsupported detection copy', async () => {
    let resolve: ((value: { registry_version: number; cameras: [] }) => void) | undefined;
    vi.mocked(fetchCameras).mockImplementation(() => new Promise((done) => { resolve = done; }));
    const host = document.createElement('div');
    document.body.append(host);
    const root = createRoot(host);
    act(() => root.render(<CameraManagementPage />));
    expect(document.body.textContent).toContain('카메라 목록을 불러오는 중');
    expect(document.body.textContent).not.toContain('카메라 목록을 새로 확인하고 있습니다.');
    await act(async () => resolve?.({ registry_version: 1, cameras: [] }));
    expect(document.body.textContent).toContain('등록된 카메라가 없습니다');
    expect(document.body.textContent).not.toContain('RTSP');
    expect(document.body.textContent).not.toContain('Registry v');
    expect(document.body.textContent).not.toMatch(/탐지 설정|설정 저장|detectionSettings/);
    act(() => root.unmount());
  });

  it('retains a successful empty registry during background refresh and clears the indicator on recovery', async () => {
    vi.useFakeTimers();
    let resolveRefresh: ((value: { registry_version: number; cameras: [] }) => void) | undefined;
    try {
      vi.mocked(fetchCameras)
        .mockResolvedValueOnce({ registry_version: 1, cameras: [] })
        .mockImplementationOnce(() => new Promise((resolve) => { resolveRefresh = resolve; }));
      const { root } = await renderPage();
      expect(document.body.textContent).toContain('등록된 카메라가 없습니다.');

      await act(async () => { await vi.advanceTimersByTimeAsync(5_000); });
      expect(document.body.textContent).toContain('카메라 목록을 새로 확인하고 있습니다.');
      expect(document.body.textContent).toContain('등록된 카메라가 없습니다.');
      expect(document.querySelectorAll('[role="status"]')).toHaveLength(1);

      await act(async () => resolveRefresh?.({ registry_version: 2, cameras: [] }));
      expect(document.body.textContent).not.toContain('카메라 목록을 새로 확인하고 있습니다.');
      expect(document.body.textContent).toContain('등록된 카메라가 없습니다.');
      expect(document.body.textContent).toContain('등록 버전 2');
      act(() => root.unmount());
    } finally {
      vi.useRealTimers();
    }
  });

  it('retains camera cards during background refresh and replaces them when recovery succeeds', async () => {
    vi.useFakeTimers();
    let resolveRefresh: ((value: { registry_version: number; cameras: Camera[] }) => void) | undefined;
    try {
      vi.mocked(fetchCameras)
        .mockResolvedValueOnce({ registry_version: 1, cameras: [camera] })
        .mockImplementationOnce(() => new Promise((resolve) => { resolveRefresh = resolve; }));
      const { root } = await renderPage();

      await act(async () => { await vi.advanceTimersByTimeAsync(5_000); });
      expect(document.body.textContent).toContain('카메라 목록을 새로 확인하고 있습니다.');
      expect(document.body.textContent).toContain('301호 A');
      expect(document.querySelectorAll('[role="status"]')).toHaveLength(1);

      await act(async () => resolveRefresh?.({ registry_version: 2, cameras: [{ ...camera, label: '301호 B' }] }));
      expect(document.body.textContent).not.toContain('카메라 목록을 새로 확인하고 있습니다.');
      expect(document.body.textContent).toContain('301호 B');
      expect(document.body.textContent).not.toContain('301호 A');
      act(() => root.unmount());
    } finally {
      vi.useRealTimers();
    }
  });

  it('shows request error and recovers through retry', async () => {
    vi.mocked(fetchCameras).mockRejectedValueOnce(new Error('offline')).mockResolvedValueOnce({ registry_version: 2, cameras: [camera] });
    const { root } = await renderPage();
    expect(document.body.textContent).toContain('카메라 목록을 불러오지 못했습니다');
    expect(document.body.textContent).not.toContain('마지막 확인');
    expect(document.body.textContent).not.toContain('등록된 카메라가 없습니다');
    await act(async () => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '다시 시도')?.click());
    expect(document.body.textContent).toContain('301호 A');
    act(() => root.unmount());
  });

  it('uses the bounded semantic camera management surface', async () => {
    vi.mocked(fetchCameras).mockResolvedValue({ registry_version: 1, cameras: [camera] });
    const { root } = await renderPage();
    const surface = document.querySelector('section');
    const classNames = document.body.innerHTML;

    expect(surface?.className).toContain('rounded-2xl');
    expect(surface?.className).toContain('border-border');
    expect(Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '+ 카메라 추가')?.classList.contains('brand-action')).toBe(true);
    expect(classNames).not.toMatch(/rounded-(3xl|4xl)|shadow-glow|shadow-soft|bg-white|text-white|slate-|rose-/);
    act(() => root.unmount());
  });

  it('does not treat a pending camera mutation as a background refresh', async () => {
    vi.mocked(fetchCameras).mockResolvedValueOnce({ registry_version: 1, cameras: [camera] });
    vi.mocked(updateCamera).mockImplementation(() => new Promise(() => {}));
    const { root } = await renderPage();

    act(() => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '수정')?.click());
    setInput('label', '301호 B');
    act(() => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '저장')?.click());

    expect(document.body.textContent).toContain('저장 중...');
    expect(document.body.textContent).not.toContain('카메라 목록을 새로 확인하고 있습니다.');
    act(() => root.unmount());
  });

  it('keeps the last successful registry visible when polling later fails', async () => {
    vi.useFakeTimers();
    try {
      const firstSuccessAt = new Date('2026-07-21T00:00:00Z');
      vi.setSystemTime(firstSuccessAt);
      vi.mocked(fetchCameras).mockResolvedValueOnce({ registry_version: 1, cameras: [camera] }).mockRejectedValueOnce(new Error('offline'));
      const { root } = await renderPage();
      expect(document.body.textContent).toContain('301호 A');
      await act(async () => { await vi.advanceTimersByTimeAsync(5_000); });
      expect(document.body.textContent).toContain('마지막 목록을 표시합니다');
      expect(document.body.textContent).not.toContain('카메라 목록을 새로 확인하고 있습니다.');
      expect(document.body.textContent).toContain(`마지막 확인 ${firstSuccessAt.toLocaleString('ko-KR')}`);
      expect(document.body.textContent).toContain('301호 A');
      act(() => root.unmount());
    } finally {
      vi.useRealTimers();
    }
  });

  it('uses the recovery success time when a later camera refresh fails', async () => {
    vi.useFakeTimers();
    try {
      const recoveredAt = new Date('2026-07-21T03:04:05Z');
      vi.mocked(fetchCameras)
        .mockRejectedValueOnce(new Error('offline'))
        .mockResolvedValueOnce({ registry_version: 2, cameras: [camera] })
        .mockRejectedValueOnce(new Error('offline again'));
      const { root } = await renderPage();
      expect(document.body.textContent).not.toContain('마지막 확인');

      vi.setSystemTime(recoveredAt);
      await act(async () => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '다시 시도')?.click());
      expect(document.body.textContent).toContain('301호 A');

      await act(async () => { await vi.advanceTimersByTimeAsync(5_000); });
      expect(document.body.textContent).toContain('마지막 목록을 표시합니다');
      expect(document.body.textContent).toContain(`마지막 확인 ${recoveredAt.toLocaleString('ko-KR')}`);
      act(() => root.unmount());
    } finally {
      vi.useRealTimers();
    }
  });

  it('retains the delete dialog and reports failure, then deletes on retry', async () => {
    vi.mocked(fetchCameras).mockResolvedValue({ registry_version: 1, cameras: [camera] });
    vi.mocked(deleteCamera).mockRejectedValueOnce(new Error('conflict')).mockResolvedValueOnce();
    const { root } = await renderPage();
    act(() => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '삭제')?.click());
    await act(async () => Array.from(document.querySelectorAll<HTMLButtonElement>('[role="dialog"] button')).find((button) => button.textContent === '삭제')?.click());
    expect(document.body.textContent).toContain('카메라 삭제에 실패했습니다');
    expect(document.querySelector('[role="dialog"]')).not.toBeNull();
    await act(async () => Array.from(document.querySelectorAll<HTMLButtonElement>('[role="dialog"] button')).find((button) => button.textContent === '삭제')?.click());
    expect(deleteCamera).toHaveBeenCalledTimes(2);
    expect(document.querySelector('[role="dialog"]')).toBeNull();
    expect(document.body.textContent).not.toContain('301호 A');
    act(() => root.unmount());
  });

  it('keeps a confirmed create when an older deferred poll completes', async () => {
    vi.useFakeTimers();
    let resolvePoll: ((value: { registry_version: number; cameras: typeof camera[] }) => void) | undefined;
    try {
      vi.mocked(fetchCameras)
        .mockResolvedValueOnce({ registry_version: 1, cameras: [camera] })
        .mockImplementationOnce(() => new Promise((resolve) => { resolvePoll = resolve; }))
        .mockResolvedValueOnce({ registry_version: 2, cameras: [camera, createdCamera] });
      vi.mocked(createCamera).mockResolvedValue(createdCamera);
      vi.mocked(testCamera).mockResolvedValue({ ok: true, width: 640, height: 480 });
      const { root } = await renderPage();
      await act(async () => { await vi.advanceTimersByTimeAsync(5_000); });

      act(() => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '+ 카메라 추가')?.click());
      setInput('label', '302호 A');
      setInput('rtspHost', '10.0.0.6');
      await act(async () => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '카메라 등록')?.click());
      expect(document.body.textContent).toContain('302호 A');

      await act(async () => resolvePoll?.({ registry_version: 1, cameras: [camera] }));
      expect(document.body.textContent).toContain('302호 A');
      act(() => root.unmount());
    } finally {
      vi.useRealTimers();
    }
  });

  it('invalidates a deferred initial load before applying a confirmed create', async () => {
    let resolveInitialLoad: ((value: { registry_version: number; cameras: typeof camera[] }) => void) | undefined;
    vi.mocked(fetchCameras)
      .mockImplementationOnce(() => new Promise((resolve) => { resolveInitialLoad = resolve; }))
      .mockResolvedValueOnce({ registry_version: 8, cameras: [camera, createdCamera] });
    vi.mocked(createCamera).mockResolvedValue(createdCamera);
    vi.mocked(testCamera).mockResolvedValue({ ok: true, width: 640, height: 480 });
    const { root } = await renderPage();

    act(() => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '+ 카메라 추가')?.click());
    setInput('label', '302호 A');
    setInput('rtspHost', '10.0.0.6');
    await act(async () => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '카메라 등록')?.click());

    expect(fetchCameras).toHaveBeenCalledTimes(2);
    expect(document.body.textContent).toContain('302호 A');
    await act(async () => resolveInitialLoad?.({ registry_version: 99, cameras: [camera] }));
    expect(document.body.textContent).toContain('302호 A');
    act(() => root.unmount());
  });

  it('keeps a confirmed update when an older deferred poll completes', async () => {
    vi.useFakeTimers();
    let resolvePoll: ((value: { registry_version: number; cameras: typeof camera[] }) => void) | undefined;
    try {
      vi.mocked(fetchCameras)
        .mockResolvedValueOnce({ registry_version: 1, cameras: [camera] })
        .mockImplementationOnce(() => new Promise((resolve) => { resolvePoll = resolve; }))
        .mockResolvedValueOnce({ registry_version: 2, cameras: [{ ...camera, label: '301호 B' }] });
      vi.mocked(updateCamera).mockResolvedValue({ ...camera, label: '301호 B' });
      const { root } = await renderPage();
      await act(async () => { await vi.advanceTimersByTimeAsync(5_000); });

      act(() => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '수정')?.click());
      setInput('label', '301호 B');
      await act(async () => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '저장')?.click());
      expect(document.body.textContent).toContain('301호 B');

      await act(async () => resolvePoll?.({ registry_version: 1, cameras: [camera] }));
      expect(document.body.textContent).toContain('301호 B');
      expect(document.body.textContent).not.toContain('301호 A');
      act(() => root.unmount());
    } finally {
      vi.useRealTimers();
    }
  });

  it('ignores an older label PATCH response after a newer decode PATCH and accepts authoritative polling', async () => {
    let resolveLabel: ((value: typeof camera) => void) | undefined;
    let resolveDecode: ((value: typeof camera) => void) | undefined;
    const pollResolvers: Array<(value: { registry_version: number; cameras: typeof camera[] }) => void> = [];
    vi.mocked(fetchCameras)
      .mockResolvedValueOnce({ registry_version: 1, cameras: [camera] })
      .mockImplementation(() => new Promise((resolve) => { pollResolvers.push(resolve); }));
    vi.mocked(updateCamera).mockImplementation(() => new Promise((resolve) => { resolveLabel = resolve; }));
    vi.mocked(updateCameraDecodeBackend).mockImplementation(() => new Promise((resolve) => { resolveDecode = resolve; }));
    const { root } = await renderPage();

    act(() => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '수정')?.click());
    setInput('label', '301호 B');
    act(() => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '저장')?.click());
    const select = document.querySelector('select');
    if (!(select instanceof HTMLSelectElement)) throw new Error('missing decode backend select');
    act(() => {
      Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, 'value')?.set?.call(select, 'cpu');
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });

    await act(async () => resolveDecode?.({ ...camera, decode_backend: 'cpu' }));
    expect((document.querySelector('select') as HTMLSelectElement).value).toBe('cpu');

    await act(async () => resolveLabel?.({ ...camera, label: '301호 B', decode_backend: 'auto' }));
    expect((document.querySelector('select') as HTMLSelectElement).value).toBe('cpu');
    expect(document.body.textContent).not.toContain('301호 B');

    await act(async () => pollResolvers.at(-1)?.({
      registry_version: 2,
      cameras: [{ ...camera, label: '301호 B', decode_backend: 'cpu' }],
    }));
    expect(document.body.textContent).toContain('301호 B');
    expect((document.querySelector('select') as HTMLSelectElement).value).toBe('cpu');
    expect(document.body.textContent).toContain('등록 버전 2');
    act(() => root.unmount());
  });

  it('ignores an older decode PATCH response after a newer label PATCH and accepts authoritative polling', async () => {
    let resolveLabel: ((value: typeof camera) => void) | undefined;
    let resolveDecode: ((value: typeof camera) => void) | undefined;
    const pollResolvers: Array<(value: { registry_version: number; cameras: typeof camera[] }) => void> = [];
    vi.mocked(fetchCameras)
      .mockResolvedValueOnce({ registry_version: 1, cameras: [camera] })
      .mockImplementation(() => new Promise((resolve) => { pollResolvers.push(resolve); }));
    vi.mocked(updateCamera).mockImplementation(() => new Promise((resolve) => { resolveLabel = resolve; }));
    vi.mocked(updateCameraDecodeBackend).mockImplementation(() => new Promise((resolve) => { resolveDecode = resolve; }));
    const { root } = await renderPage();

    const select = document.querySelector('select');
    if (!(select instanceof HTMLSelectElement)) throw new Error('missing decode backend select');
    act(() => {
      Object.getOwnPropertyDescriptor(window.HTMLSelectElement.prototype, 'value')?.set?.call(select, 'cpu');
      select.dispatchEvent(new Event('change', { bubbles: true }));
    });
    act(() => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '수정')?.click());
    setInput('label', '301호 B');
    act(() => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '저장')?.click());

    await act(async () => resolveLabel?.({ ...camera, label: '301호 B', decode_backend: 'auto' }));
    expect(document.body.textContent).toContain('301호 B');

    await act(async () => resolveDecode?.({ ...camera, label: '301호 A', decode_backend: 'cpu' }));
    expect(document.body.textContent).toContain('301호 B');
    expect(document.body.textContent).not.toContain('301호 A');

    await act(async () => pollResolvers.at(-1)?.({
      registry_version: 2,
      cameras: [{ ...camera, label: '301호 B', decode_backend: 'cpu' }],
    }));
    expect(document.body.textContent).toContain('301호 B');
    expect((document.querySelector('select') as HTMLSelectElement).value).toBe('cpu');
    expect(document.body.textContent).toContain('등록 버전 2');
    act(() => root.unmount());
  });

  it('does not resurrect a confirmed delete when an older deferred poll completes', async () => {
    vi.useFakeTimers();
    let resolvePoll: ((value: { registry_version: number; cameras: typeof camera[] }) => void) | undefined;
    try {
      vi.mocked(fetchCameras)
        .mockResolvedValueOnce({ registry_version: 1, cameras: [camera] })
        .mockImplementationOnce(() => new Promise((resolve) => { resolvePoll = resolve; }))
        .mockResolvedValueOnce({ registry_version: 2, cameras: [] });
      vi.mocked(deleteCamera).mockResolvedValue();
      const { root } = await renderPage();
      await act(async () => { await vi.advanceTimersByTimeAsync(5_000); });

      act(() => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '삭제')?.click());
      await act(async () => Array.from(document.querySelectorAll<HTMLButtonElement>('[role="dialog"] button')).find((button) => button.textContent === '삭제')?.click());
      expect(document.body.textContent).not.toContain('301호 A');

      await act(async () => resolvePoll?.({ registry_version: 1, cameras: [camera] }));
      expect(document.body.textContent).not.toContain('301호 A');
      act(() => root.unmount());
    } finally {
      vi.useRealTimers();
    }
  });

  it('disables delete while an update for the same camera is pending, then allows it once the update settles', async () => {
    let resolveUpdate: ((value: typeof camera) => void) | undefined;
    vi.mocked(fetchCameras).mockResolvedValue({ registry_version: 4, cameras: [camera] });
    vi.mocked(updateCamera).mockImplementation(() => new Promise((resolve) => { resolveUpdate = resolve; }));
    vi.mocked(deleteCamera).mockResolvedValue();
    const { root } = await renderPage();

    act(() => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '수정')?.click());
    setInput('label', '301호 B');
    act(() => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '저장')?.click());

    const deleteButton = Array.from(document.querySelectorAll<HTMLButtonElement>('button')).find((button) => button.textContent === '삭제');
    // The card's own delete button is disabled while its rename is still in flight, consistent
    // with the other per-card mutation buttons — this prevents deleting a camera whose edit
    // hasn't resolved yet.
    expect(deleteButton?.disabled).toBe(true);
    act(() => deleteButton?.click());
    expect(document.body.textContent).not.toContain('삭제 확인');

    await act(async () => resolveUpdate?.({ ...camera, label: '301호 B' }));
    expect(deleteButton?.disabled).toBe(false);

    act(() => deleteButton?.click());
    await act(async () => Array.from(document.querySelectorAll<HTMLButtonElement>('[role="dialog"] button')).find((button) => button.textContent === '삭제')?.click());

    expect(deleteCamera).toHaveBeenCalledWith('cam-1');
    expect(document.body.textContent).not.toContain('301호 B');
    act(() => root.unmount());
  });

  it('accepts a same-id recreation and later polls after a newer authoritative registry version', async () => {
    vi.useFakeTimers();
    let resolveDeleteRefresh: ((value: { registry_version: number; cameras: typeof camera[] }) => void) | undefined;
    const recreated = { ...camera, label: '301호 재등록' };
    try {
      vi.mocked(fetchCameras)
        .mockResolvedValueOnce({ registry_version: 4, cameras: [camera] })
        .mockImplementationOnce(() => new Promise((resolve) => { resolveDeleteRefresh = resolve; }))
        .mockResolvedValueOnce({ registry_version: 6, cameras: [{ ...recreated, label: '301호 재등록 갱신' }] });
      vi.mocked(deleteCamera).mockResolvedValue();
      const { root } = await renderPage();

      act(() => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '삭제')?.click());
      await act(async () => Array.from(document.querySelectorAll<HTMLButtonElement>('[role="dialog"] button')).find((button) => button.textContent === '삭제')?.click());
      expect(document.body.textContent).not.toContain('301호 A');

      await act(async () => resolveDeleteRefresh?.({ registry_version: 5, cameras: [recreated] }));
      expect(document.body.textContent).toContain('301호 재등록');

      await act(async () => { await vi.advanceTimersByTimeAsync(5_000); });
      expect(document.body.textContent).toContain('301호 재등록 갱신');
      expect(document.body.textContent).toContain('등록 버전 6');
      act(() => root.unmount());
    } finally {
      vi.useRealTimers();
    }
  });

  it('warns on a duplicate label (trimmed, case-insensitive) without disabling the add-camera submit button', async () => {
    // R2.2: a label match against an already-registered camera is a non-blocking warning. This
    // proves the roster actually reaches AddCameraModal via the `cameras={registry.cameras}`
    // wiring in CameraManagementPage — not just that the comparison function works in isolation.
    const existingCamera = { ...camera, label: '301호 침대 A' };
    vi.mocked(fetchCameras).mockResolvedValue({ registry_version: 1, cameras: [existingCamera] });
    const { root } = await renderPage();

    act(() => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '+ 카메라 추가')?.click());
    // Trims whitespace and differs in case from the registered label, exercising the real
    // findDuplicateLabelCamera comparison rather than a trivial exact match.
    setInput('label', '  301호 침대 a  ');

    const warning = Array.from(document.querySelectorAll('[role="status"]')).find((el) => el.textContent?.includes('이름이 같습니다'));
    expect(warning).toBeDefined();
    expect(warning?.textContent).toContain('301호 침대 A');

    const submitButton = Array.from(document.querySelectorAll<HTMLButtonElement>('button')).find((button) => button.textContent === '카메라 등록');
    expect(submitButton?.disabled).toBe(false);
    act(() => root.unmount());
  });

  it('reconciles stale and newer roster identities emitted through a backend camera alias', async () => {
    vi.useFakeTimers();
    const mappedCamera = { ...camera, backend_camera_id: 'backend-cam-1' };
    const aliasRosterCamera = { ...camera, id: 'backend-cam-1', backend_camera_id: null, label: '서버 재등록 카메라' };
    let resolveStalePoll: ((value: { registry_version: number; cameras: typeof aliasRosterCamera[] }) => void) | undefined;
    try {
      vi.mocked(fetchCameras)
        .mockResolvedValueOnce({ registry_version: 7, cameras: [mappedCamera] })
        .mockImplementationOnce(() => new Promise((resolve) => { resolveStalePoll = resolve; }))
        .mockResolvedValueOnce({ registry_version: 8, cameras: [aliasRosterCamera] });
      vi.mocked(deleteCamera).mockResolvedValue();
      const { root } = await renderPage();

      act(() => Array.from(document.querySelectorAll('button')).find((button) => button.textContent === '삭제')?.click());
      await act(async () => Array.from(document.querySelectorAll<HTMLButtonElement>('[role="dialog"] button')).find((button) => button.textContent === '삭제')?.click());
      await act(async () => resolveStalePoll?.({ registry_version: 7, cameras: [aliasRosterCamera] }));
      expect(document.body.textContent).not.toContain('서버 재등록 카메라');

      await act(async () => { await vi.advanceTimersByTimeAsync(5_000); });
      expect(deleteCamera).toHaveBeenCalledWith('cam-1');
      expect(document.body.textContent).toContain('서버 재등록 카메라');
      expect(document.body.textContent).toContain('등록 버전 8');
      act(() => root.unmount());
    } finally {
      vi.useRealTimers();
    }
  });
});
