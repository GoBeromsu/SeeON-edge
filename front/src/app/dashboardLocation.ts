export const DASHBOARD_QUERY_KEYS = ['page', 'floor', 'camera', 'event', 'clip'] as const;
export const DASHBOARD_PAGES = ['operations', 'events', 'settings'] as const;

export type DashboardPage = (typeof DASHBOARD_PAGES)[number];
export type DashboardLocation = {
  page: DashboardPage;
  floor?: string;
  camera?: string;
  event?: string;
  clip?: string;
};

type Resource<T> = {
  status: 'idle' | 'loading' | 'success' | 'error';
  data?: readonly T[];
  eventTypes?: readonly string[];
  complete?: boolean;
};

type LocationCamera = {
  id: string;
  floor_name?: string | null;
};

type LocationClip = {
  id: string;
  event_type: string;
};

export type DashboardLocationData = {
  cameras?: Resource<LocationCamera>;
  clips?: Resource<LocationClip>;
};

export type CanonicalDashboardLocation = {
  location: DashboardLocation;
  search: string;
  changed: boolean;
};

type RestorationPhase = 'syntax' | 'data' | null;

function lastValue(params: URLSearchParams, key: string): string | undefined {
  const values = params.getAll(key);
  const value = values.at(-1);
  return value ? value : undefined;
}

function parseSyntax(search: string): DashboardLocation {
  const params = new URLSearchParams(search);
  const rawPage = lastValue(params, 'page');
  const page = DASHBOARD_PAGES.includes(rawPage as DashboardPage) ? rawPage as DashboardPage : 'operations';
  const location: DashboardLocation = { page };

  if (page === 'operations') {
    for (const key of ['floor', 'camera'] as const) {
      const value = lastValue(params, key);
      if (value !== undefined) location[key] = value;
    }
  }

  if (page === 'events') {
    for (const key of ['event', 'clip'] as const) {
      const value = lastValue(params, key);
      if (value !== undefined) location[key] = value;
    }
  }

  return location;
}

function validateOperations(location: DashboardLocation, cameras: readonly LocationCamera[]): void {
  if (location.floor && !cameras.some((camera) => camera.floor_name === location.floor)) {
    delete location.floor;
  }
  if (location.camera) {
    const match = cameras.find((camera) => camera.id === location.camera);
    if (!match || (location.floor && match.floor_name !== location.floor)) {
      delete location.camera;
    }
  }
}

function validateEvents(location: DashboardLocation, clips: Resource<LocationClip>): void {
  const data = clips.data ?? [];
  const eventTypes = clips.eventTypes ?? data.map((clip) => clip.event_type);
  if (location.event && !eventTypes.includes(location.event)) {
    delete location.event;
  }
  if (location.clip && clips.complete !== false) {
    const match = data.find((clip) => clip.id === location.clip);
    if (!match || (location.event && match.event_type !== location.event)) {
      delete location.clip;
    }
  }
}

export function serializeDashboardLocation(location: DashboardLocation): string {
  const params = new URLSearchParams();
  for (const key of DASHBOARD_QUERY_KEYS) {
    const value = location[key];
    if (value !== undefined) params.append(key, value);
  }
  return `?${params.toString()}`;
}

export function canonicalizeDashboardLocation(search: string, data: DashboardLocationData = {}): CanonicalDashboardLocation {
  const location = parseSyntax(search);
  if (location.page === 'operations' && data.cameras?.status === 'success') {
    validateOperations(location, data.cameras.data ?? []);
  }
  if (location.page === 'events' && data.clips?.status === 'success') {
    validateEvents(location, data.clips);
  }
  const canonicalSearch = serializeDashboardLocation(location);
  return { location, search: canonicalSearch, changed: canonicalSearch !== search };
}

function needsDataValidation(location: DashboardLocation): boolean {
  return location.page === 'operations' || location.page === 'events';
}

function hasRelevantData(location: DashboardLocation, data: DashboardLocationData | undefined): boolean {
  if (location.page === 'operations') return data?.cameras?.status === 'success';
  if (location.page === 'events') return data?.clips?.status === 'success';
  return false;
}

export class DashboardLocationController {
  private restorationPhase: RestorationPhase = null;
  private readonly onPopState = (): void => {
    this.restorationPhase = 'syntax';
    this.canonicalize();
  };

  constructor(
    private readonly target: Window,
    private readonly onChange: (location: DashboardLocation) => void = () => undefined,
  ) {}

  start(data?: DashboardLocationData): DashboardLocation {
    this.target.addEventListener('popstate', this.onPopState);
    return this.canonicalize(data);
  }

  canonicalize(data?: DashboardLocationData): DashboardLocation {
    const result = canonicalizeDashboardLocation(this.target.location.search, data);
    const phase = this.restorationPhase;
    const relevantData = hasRelevantData(result.location, data);
    const replacementAvailable = phase === null || phase === 'syntax' || relevantData;
    if (result.changed && replacementAvailable) {
      this.target.history.replaceState(null, '', `${this.target.location.pathname}${result.search}${this.target.location.hash}`);
    }
    if (phase === 'syntax') this.restorationPhase = needsDataValidation(result.location) ? 'data' : null;
    else if (phase === 'data' && relevantData) this.restorationPhase = null;
    this.onChange(result.location);
    return result.location;
  }

  navigate(update: Partial<Record<keyof DashboardLocation, string | null | undefined>>): DashboardLocation {
    this.restorationPhase = null;
    const current = canonicalizeDashboardLocation(this.target.location.search).location;
    const next = { ...current } as Record<string, string | undefined>;
    for (const [key, value] of Object.entries(update)) {
      if (value === null || value === undefined) delete next[key];
      else next[key] = value;
    }
    const result = canonicalizeDashboardLocation(serializeDashboardLocation(next as DashboardLocation));
    if (result.search !== this.target.location.search) {
      this.target.history.pushState(null, '', `${this.target.location.pathname}${result.search}${this.target.location.hash}`);
    }
    this.onChange(result.location);
    return result.location;
  }

  reset(): DashboardLocation {
    this.restorationPhase = null;
    const result = canonicalizeDashboardLocation('');
    if (result.search !== this.target.location.search) {
      this.target.history.replaceState(null, '', `${this.target.location.pathname}${result.search}${this.target.location.hash}`);
    }
    this.onChange(result.location);
    return result.location;
  }

  dispose(): void {
    this.target.removeEventListener('popstate', this.onPopState);
  }
}
