import { act } from 'react';
import { createRoot, type Root } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { confirmTopologyPreview, fetchTopologyPreview, type CameraTopology, type TopologyPreview } from '@/shared/api/topologyClient';
import type { CameraRegistry } from '@/shared/api/client';
import { ServerSyncStep } from '@/features/connection/ServerSyncStep';

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
  cameras: 1,
  rooms: 0,
  floors: 0,
  confirmed: false,
};

function topologyWithMapped(count: number): CameraTopology {
  return {
    registry_version: 1,
    dirty_registry_version: null,
    readiness_error: null,
    unmapped_camera_ids: [],
    floors: [{
      edge_ref: 'floor-1', name: '1층', order_index: 1,
      rooms: [{
        edge_ref: 'room-1', name: '101호', room_type: 'ROOM', capacity: 1, legacy_canonical_space_id: null,
        cameras: Array.from({ length: count }, (_, index) => ({ edge_ref: `cam-${index}`, label: `cam-${index}` })),
      }],
    }],
  };
}

function cameras(count: number): CameraRegistry {
  return { registry_version: 1, cameras: Array.from({ length: count }, (_, index) => ({ id: `cam-${index}`, label: `cam-${index}`, rtsp_url_masked: 'x', floor_name: null, status: 'online', created_at: null })) };
}

function button(scope: ParentNode, label: string): HTMLButtonElement {
  const match = Array.from(scope.querySelectorAll('button')).find((candidate) => candidate.textContent?.trim() === label);
  if (!match) throw new Error(`missing button ${label}`);
  return match;
}

async function renderStep(props: { readonly cameras: CameraRegistry; readonly topology: CameraTopology; readonly preview: TopologyPreview | null; readonly onChanged?: () => Promise<void> }): Promise<{ readonly host: HTMLDivElement; readonly root: Root }> {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  await act(async () => {
    root.render(
      <ServerSyncStep
        cameras={props.cameras}
        topology={props.topology}
        preview={props.preview}
        onPreviewChanged={() => undefined}
        onChanged={props.onChanged ?? (() => Promise.resolve())}
      />,
    );
  });
  return { host, root };
}

beforeEach(() => {
  vi.mocked(fetchTopologyPreview).mockResolvedValue({ preview });
  vi.mocked(confirmTopologyPreview).mockResolvedValue({ snapshot_id: preview.snapshot_id, client_revision: 4, server_revision: 10 });
});

afterEach(() => {
  document.body.innerHTML = '';
  vi.clearAllMocks();
});

describe('ServerSyncStep', () => {
  it('reports "already up to date" when nothing is pending, without a confirm action', async () => {
    const { host } = await renderStep({ cameras: cameras(1), topology: topologyWithMapped(1), preview: null });

    expect(host.textContent).toContain('이미 최신 상태입니다');
    expect(host.querySelectorAll('button')).toHaveLength(0);
  });

  it('surfaces a pending preview as a warning requiring explicit confirmation, using the deactivation copy verbatim', async () => {
    const { host } = await renderStep({ cameras: cameras(1), topology: topologyWithMapped(1), preview });

    expect(host.textContent).toContain('삭제되지 않고 비활성화됩니다');
    expect(host.querySelector('[role="alert"]')).toBeNull();
    expect(() => button(host, '변경 사항 확인 후 반영')).not.toThrow();
  });

  it('reports an already-confirmed preview distinctly from a pending one', async () => {
    const { host } = await renderStep({ cameras: cameras(1), topology: topologyWithMapped(1), preview: { ...preview, confirmed: true } });

    expect(host.textContent).toContain('이미 확정했습니다');
  });

  it('warns (does not silently confirm) when the mapped camera count drifts from the edge registry total -- last line of defense', async () => {
    const { host } = await renderStep({ cameras: cameras(3), topology: topologyWithMapped(2), preview: null });

    expect(host.querySelector('[role="alert"]')?.textContent).toContain('다릅니다');
  });

  it('does not warn about a mismatch when the mapped total matches the edge registry total', async () => {
    const { host } = await renderStep({ cameras: cameras(2), topology: topologyWithMapped(2), preview: null });

    expect(host.querySelector('[role="alert"]')).toBeNull();
  });

  it('confirms through the shared dialog and reloads via onChanged', async () => {
    const onChanged = vi.fn().mockResolvedValue(undefined);
    const { host } = await renderStep({ cameras: cameras(1), topology: topologyWithMapped(1), preview, onChanged });
    act(() => button(host, '변경 사항 확인 후 반영').click());
    const dialog = document.querySelector('[role="dialog"]');
    if (!dialog) throw new Error('missing dialog');

    await act(async () => button(dialog, '제외 확정').click());

    expect(confirmTopologyPreview).toHaveBeenCalled();
    expect(onChanged).toHaveBeenCalled();
  });
});
