import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { SnapshotQueue } from '@/features/operations/SnapshotQueue';

const identity = [{ id: 'tile-cam-1', mediaId: 'cam-1' }] as const;

describe('SnapshotQueue', () => {
  beforeEach(() => {
    vi.useFakeTimers();
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  it('does not request another snapshot after the identity settles while activation remains enabled', async () => {
    // Given one active camera identity and its initial snapshot request.
    const urlFor = vi.fn((mediaId: string, refreshKey: number) => `/snapshot/${mediaId}?refresh=${refreshKey}`);
    const queue = new SnapshotQueue(urlFor);
    queue.update(identity, true);
    const requestUrl = queue.get(identity[0].id)?.requestUrl;
    expect(requestUrl).toBe('/snapshot/cam-1?refresh=0');

    // When that request resolves and far more than the old refresh window elapses.
    queue.resolve(identity[0].id, identity[0].mediaId, 'loaded', requestUrl ?? undefined);
    await vi.advanceTimersByTimeAsync(60_000);

    // Then the identity remains settled and no recurring request is created.
    expect(urlFor).toHaveBeenCalledTimes(1);
    expect(queue.get(identity[0].id)).toMatchObject({ state: 'loaded', requestUrl: null });
    queue.dispose();
  });

  it('requests one new snapshot when the same identity is deactivated and activated again', () => {
    // Given one identity whose first activation has settled.
    const urlFor = vi.fn((mediaId: string, refreshKey: number) => `/snapshot/${mediaId}?refresh=${refreshKey}`);
    const queue = new SnapshotQueue(urlFor);
    queue.update(identity, true);
    const firstRequestUrl = queue.get(identity[0].id)?.requestUrl;
    queue.resolve(identity[0].id, identity[0].mediaId, 'loaded', firstRequestUrl ?? undefined);

    // When the wall deactivates and later activates that identity again.
    queue.update(identity, false);
    queue.update(identity, true);
    queue.update(identity, true);

    // Then exactly one new activation request is issued with a new cache key.
    expect(urlFor).toHaveBeenCalledTimes(2);
    expect(queue.get(identity[0].id)?.requestUrl).toBe('/snapshot/cam-1?refresh=1');
    queue.dispose();
  });
});
