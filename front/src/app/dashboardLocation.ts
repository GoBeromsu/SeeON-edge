export const DASHBOARD_QUERY_KEYS = ['page', 'floor', 'room', 'camera', 'event', 'clip', 'wallPage'] as const;
export const DASHBOARD_PAGES = ['operations', 'events', 'cameras', 'system'] as const;

export type DashboardPage = (typeof DASHBOARD_PAGES)[number];
export type DashboardLocation = {
  page: DashboardPage;
  floor?: string;
  room?: string;
  camera?: string;
  event?: string;
  clip?: string;
  wallPage?: string;
};

type Resource<T> = {
  status: 'idle' | 'loading' | 'success' | 'error';
  data?: readonly T[];
};

type LocationCamera = {
  id: string;
  backend_camera_id?: string | null;
  floor_name?: string | null;
  space_id?: string | null;
};

type LocationClip = {
  id: string;
  camera_id: string | null;
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

const MAX_WALL_PAGE = 9_007_199_254_740_991n;
const WALL_PAGE_SIZE = 12n;

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

  if (page === 'operations' || page === 'events') {
    for (const key of ['floor', 'room', 'camera'] as const) {
      const value = lastValue(params, key);
      if (value !== undefined) location[key] = value;
    }
  }

  if (page === 'operations') {
    const wallPage = lastValue(params, 'wallPage');
    location.wallPage = wallPage && /^[1-9][0-9]*$/.test(wallPage) && BigInt(wallPage) <= MAX_WALL_PAGE ? wallPage : '1';
  }

  if (page === 'events') {
    const event = lastValue(params, 'event');
    const clip = lastValue(params, 'clip');
    if (event !== undefined) location.event = event;
    if (clip !== undefined) location.clip = clip;
  }

  return location;
}

function cameraMatchesLocation(camera: LocationCamera, location: DashboardLocation): boolean {
  return (!location.floor || camera.floor_name === location.floor)
    && (!location.room || camera.space_id === location.room);
}

function validateCameraFilters(location: DashboardLocation, cameras: readonly LocationCamera[]): void {
  if (location.floor && !cameras.some((camera) => camera.floor_name === location.floor)) {
    delete location.floor;
    delete location.room;
  }
  if (location.room && !cameras.some((camera) => camera.space_id === location.room && (!location.floor || camera.floor_name === location.floor))) {
    delete location.room;
  }
  if (location.camera && !cameras.some((camera) => camera.id === location.camera && cameraMatchesLocation(camera, location))) {
    delete location.camera;
  }
}

function resolveClipCamera(clip: LocationClip, cameras: readonly LocationCamera[]): LocationCamera | undefined {
  const local = cameras.find((camera) => camera.id === clip.camera_id);
  if (local) return local;
  const aliases = cameras.filter((camera) => camera.backend_camera_id !== null && camera.backend_camera_id === clip.camera_id);
  return aliases.length === 1 ? aliases[0] : undefined;
}

function validateOperations(location: DashboardLocation, cameras: readonly LocationCamera[]): void {
  validateCameraFilters(location, cameras);

  const visibleCount = BigInt(cameras.filter((camera) => cameraMatchesLocation(camera, location)).length);
  const lastPage = visibleCount === 0n ? 1n : (visibleCount + WALL_PAGE_SIZE - 1n) / WALL_PAGE_SIZE;
  if (BigInt(location.wallPage ?? '1') > lastPage) location.wallPage = lastPage.toString();
}

function validateEvents(location: DashboardLocation, cameras: readonly LocationCamera[], clips: readonly LocationClip[]): void {
  validateCameraFilters(location, cameras);

  const matchesFilters = (clip: LocationClip): boolean => {
    const camera = resolveClipCamera(clip, cameras);
    if (!camera) return !location.floor && !location.room && !location.camera;
    return cameraMatchesLocation(camera, location) && (!location.camera || camera.id === location.camera);
  };
  if (location.event && !clips.some((clip) => matchesFilters(clip) && clip.event_type === location.event)) {
    delete location.event;
  }
  if (location.clip) {
    const selected = clips.find((clip) => clip.id === location.clip);
    if (!selected || !matchesFilters(selected) || (location.event && selected.event_type !== location.event)) delete location.clip;
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
  if (location.page === 'events' && data.cameras?.status === 'success' && data.clips?.status === 'success') {
    validateEvents(location, data.cameras.data ?? [], data.clips.data ?? []);
  }
  const canonicalSearch = serializeDashboardLocation(location);
  return { location, search: canonicalSearch, changed: canonicalSearch !== search };
}

function needsDataValidation(location: DashboardLocation): boolean {
  return location.page === 'operations' || location.page === 'events';
}

function hasRelevantData(location: DashboardLocation, data: DashboardLocationData | undefined): boolean {
  if (location.page === 'operations') return data?.cameras?.status === 'success';
  if (location.page === 'events') return data?.cameras?.status === 'success' && data.clips?.status === 'success';
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
