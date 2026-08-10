import { afterEach, describe, expect, it, vi } from 'vitest';
import {
  confirmTopologyPreview,
  createTopologyFloor,
  createTopologyRoom,
  fetchTopology,
  fetchTopologyPreview,
  syncTopology,
  updateTopologyFloor,
  updateTopologyRoom,
} from '@/shared/api/topologyClient';

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe('topology API seam', () => {
  it('parses the redacted topology and omission preview without transport fields', async () => {
    const fetchMock = vi.fn()
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({
        registry_version: 4, dirty_registry_version: null, readiness_error: null, unmapped_camera_ids: [],
        floors: [{ edge_ref: 'floor-1', name: '1층', order_index: 1, rooms: [{ edge_ref: 'room-101', name: '101호', room_type: 'ROOM', capacity: 1, legacy_canonical_space_id: null, cameras: [] }] }],
      }) })
      .mockResolvedValueOnce({ ok: true, status: 200, json: async () => ({ preview: {
        confirmation_id: '0197f671-3a31-7a6c-a6e4-83ed412de81b', digest: 'a'.repeat(64),
        expires_at: '2099-01-01T00:00:00.000Z', snapshot_id: '0197f671-3a31-7a6c-a6e4-83ed412de81c',
        client_revision: 2, server_revision: 5, cameras: 1, rooms: 0, floors: 0, confirmed: false,
      } }) });
    vi.stubGlobal('fetch', fetchMock);

    await expect(fetchTopology()).resolves.toMatchObject({ registry_version: 4, floors: [{ edge_ref: 'floor-1' }] });
    await expect(fetchTopologyPreview()).resolves.toMatchObject({ preview: { cameras: 1, client_revision: 2, server_revision: 5 } });
  });

  it('serializes floor and room mutations only through authenticated local routes', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({}) });
    vi.stubGlobal('fetch', fetchMock);

    await createTopologyFloor({ edge_ref: 'floor-1', name: '1층', order_index: 1 });
    await updateTopologyFloor('floor/1', { name: '본관 1층', order_index: 2 });
    await createTopologyRoom({ edge_ref: 'room-101', floor_edge_ref: 'floor-1', name: '101호', legacy_canonical_space_id: null });
    await updateTopologyRoom('room/101', '101호 A');

    expect(fetchMock).toHaveBeenNthCalledWith(1, '/api/v1/cameras/topology/floors', expect.objectContaining({ credentials: 'same-origin', body: JSON.stringify({ edge_ref: 'floor-1', name: '1층', order_index: 1 }) }));
    expect(fetchMock).toHaveBeenNthCalledWith(2, '/api/v1/cameras/topology/floors/floor%2F1', expect.objectContaining({ method: 'PATCH' }));
    expect(fetchMock).toHaveBeenNthCalledWith(3, '/api/v1/cameras/topology/rooms', expect.objectContaining({ body: JSON.stringify({ edge_ref: 'room-101', floor_edge_ref: 'floor-1', name: '101호', legacy_canonical_space_id: null }) }));
    expect(fetchMock).toHaveBeenNthCalledWith(4, '/api/v1/cameras/topology/rooms/room%2F101', expect.objectContaining({ method: 'PATCH' }));
  });

  it('sends only the frozen preview identity to the server-side confirmation route', async () => {
    const fetchMock = vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({
      snapshot_id: '0197f671-3a31-7a6c-a6e4-83ed412de81c', client_revision: 2, server_revision: 6,
      result: { floors: {}, rooms: {}, cameras: {} },
    }) });
    vi.stubGlobal('fetch', fetchMock);
    const input = { confirmation_id: '0197f671-3a31-7a6c-a6e4-83ed412de81b', digest: 'a'.repeat(64), client_revision: 2, server_revision: 5 };

    await confirmTopologyPreview(input);

    expect(fetchMock).toHaveBeenCalledWith('/api/v1/connection/topology-preview/confirm', expect.objectContaining({
      method: 'POST', credentials: 'same-origin', body: JSON.stringify(input),
    }));
    expect(fetchMock.mock.calls[0]?.[1]?.body).not.toContain('token');
  });

  it('normalizes explicit sync conflicts without exposing backend detail bodies', async () => {
    vi.stubGlobal('fetch', vi.fn().mockResolvedValue({ ok: true, status: 200, json: async () => ({
      status: 'failed', error_class: 'conflict', detail: null, last_ok_at: null, next_retry_at: null, camera_count: 3,
    }) }));

    await expect(syncTopology()).resolves.toEqual({
      status: 'failed', error_class: 'conflict', detail: null, last_ok_at: null, next_retry_at: null, camera_count: 3,
    });
  });
});
