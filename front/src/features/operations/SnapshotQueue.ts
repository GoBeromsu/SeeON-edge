import { snapshotJitterMs } from '@/features/operations/operationsModel';

export type SnapshotLifecycle = 'loading' | 'loaded' | 'stale' | 'error';
export type SnapshotEntry = {
  id: string;
  mediaId: string;
  state: SnapshotLifecycle;
  requestUrl: string | null;
  lastLoadedUrl: string | null;
  lastLoadedAt: number | null;
};

export type SnapshotIdentity = { id: string; mediaId: string };

type InternalEntry = SnapshotEntry & { active: boolean; refreshKey: number; refreshTimer: number | null; requestTimer: number | null };

/**
 * Concurrency-capped, jittered poller for the camera-wall thumbnails: only `limit` snapshot
 * requests are ever in flight, each in-flight request times out into an `error` state after 10s,
 * and a settled entry (loaded or errored) re-queues after 5s + a per-camera jitter so the whole
 * wall doesn't refresh in lockstep. `useSyncExternalStore`-friendly via subscribe/snapshot.
 */
export class SnapshotQueue {
  private entries = new Map<string, InternalEntry>();
  private order: string[] = [];
  private enabled = false;
  private listeners = new Set<() => void>();
  private currentSnapshot: SnapshotEntry[] = [];

  constructor(private readonly urlFor: (mediaId: string, refreshKey: number) => string, private readonly limit = 6) {}

  get unresolvedCount(): number {
    return [...this.entries.values()].filter((entry) => entry.active).length;
  }

  subscribe = (listener: () => void): (() => void) => {
    this.listeners.add(listener);
    return () => this.listeners.delete(listener);
  };

  snapshot = (): SnapshotEntry[] => this.currentSnapshot;

  get(id: string): SnapshotEntry | undefined {
    return this.entries.get(id);
  }

  update(identities: readonly SnapshotIdentity[], enabled: boolean): void {
    this.enabled = enabled;
    const ids = identities.map(({ id }) => id);
    const nextIds = new Set(ids);
    for (const [id, entry] of this.entries) {
      if (!nextIds.has(id)) {
        if (entry.refreshTimer !== null) window.clearTimeout(entry.refreshTimer);
        if (entry.requestTimer !== null) window.clearTimeout(entry.requestTimer);
        this.entries.delete(id);
      }
    }
    this.order = [...ids];
    for (const { id, mediaId } of identities) {
      const current = this.entries.get(id);
      if (current?.mediaId === mediaId) continue;
      if (current && current.refreshTimer !== null) window.clearTimeout(current.refreshTimer);
      if (current && current.requestTimer !== null) window.clearTimeout(current.requestTimer);
      this.entries.set(id, { id, mediaId, state: 'loading', requestUrl: null, lastLoadedUrl: null, lastLoadedAt: null, active: false, refreshKey: 0, refreshTimer: null, requestTimer: null });
    }
    if (!enabled) {
      for (const entry of this.entries.values()) {
        if (entry.refreshTimer !== null) window.clearTimeout(entry.refreshTimer);
        if (entry.requestTimer !== null) window.clearTimeout(entry.requestTimer);
        entry.refreshTimer = null;
        entry.requestTimer = null;
        entry.active = false;
        entry.requestUrl = null;
        if (entry.lastLoadedUrl) entry.state = 'stale';
      }
    } else {
      this.drain();
    }
    this.emit();
  }

  resolve(id: string, mediaId: string, result: 'loaded' | 'error', requestUrl?: string): void {
    const entry = this.entries.get(id);
    if (!entry?.active || entry.mediaId !== mediaId || (requestUrl && entry.requestUrl !== requestUrl)) return;
    const completedUrl = entry.requestUrl;
    if (entry.requestTimer !== null) window.clearTimeout(entry.requestTimer);
    entry.requestTimer = null;
    entry.active = false;
    entry.requestUrl = null;
    if (result === 'loaded') {
      entry.state = 'loaded';
      entry.lastLoadedUrl = completedUrl;
      entry.lastLoadedAt = Date.now();
    } else {
      entry.state = 'error';
    }
    entry.refreshTimer = window.setTimeout(() => {
      entry.refreshTimer = null;
      if (!this.enabled || this.entries.get(id)?.mediaId !== mediaId) return;
      entry.refreshKey += 1;
      entry.state = entry.lastLoadedUrl ? 'stale' : 'loading';
      this.drain();
      this.emit();
    }, 5_000 + snapshotJitterMs(id));
    this.drain();
    this.emit();
  }

  dispose(): void {
    this.update([], false);
    this.listeners.clear();
  }

  private drain(): void {
    if (!this.enabled) return;
    let available = this.limit - this.unresolvedCount;
    if (available <= 0) return;
    for (const id of this.order) {
      const entry = this.entries.get(id);
      if (!entry || entry.active || entry.refreshTimer !== null || entry.requestUrl !== null) continue;
      entry.active = true;
      entry.requestUrl = this.urlFor(entry.mediaId, entry.refreshKey);
      if (!entry.lastLoadedUrl) entry.state = 'loading';
      const requestUrl = entry.requestUrl;
      entry.requestTimer = window.setTimeout(() => this.resolve(id, entry.mediaId, 'error', requestUrl), 10_000);
      if (--available === 0) break;
    }
  }

  private emit(): void {
    this.currentSnapshot = this.order.flatMap((id) => {
      const entry = this.entries.get(id);
      return entry ? [{ id: entry.id, mediaId: entry.mediaId, state: entry.state, requestUrl: entry.requestUrl, lastLoadedUrl: entry.lastLoadedUrl, lastLoadedAt: entry.lastLoadedAt }] : [];
    });
    this.listeners.forEach((listener) => listener());
  }
}
