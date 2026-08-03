import type { DetectionDomainKey, DetectionDomainSetting, DetectionMode, DetectionSettings } from '@/shared/api/types';

/** Display order matches front/design-handoff/README.md §5's read-mode example (침대 이탈 먼저, 낙상 다음). */
export const DOMAIN_ORDER: readonly DetectionDomainKey[] = ['bed_exit', 'fall'];

export const DOMAIN_LABELS: Record<DetectionDomainKey, string> = {
  bed_exit: '침대 이탈',
  fall: '낙상',
};

const DEFAULT_WINDOW_START = '21:00';
const DEFAULT_WINDOW_END = '06:00';

export function formatDomainSchedule(setting: DetectionDomainSetting): string {
  if (setting.mode === 'always') return '항상';
  return `${setting.start ?? '--:--'}–${setting.end ?? '--:--'}`;
}

function updateDomain(
  settings: DetectionSettings,
  domain: DetectionDomainKey,
  patch: Partial<DetectionDomainSetting>,
): DetectionSettings {
  return {
    domains: {
      ...settings.domains,
      [domain]: { ...settings.domains[domain], ...patch },
    },
  };
}

export function updateDomainOn(settings: DetectionSettings, domain: DetectionDomainKey, on: boolean): DetectionSettings {
  return updateDomain(settings, domain, { on });
}

/** Switching to "window" fills in a sensible night-shift default when no time is set yet; switching back to "always" clears both. */
export function updateDomainMode(settings: DetectionSettings, domain: DetectionDomainKey, mode: DetectionMode): DetectionSettings {
  if (mode === 'always') {
    return updateDomain(settings, domain, { mode, start: null, end: null });
  }
  const current = settings.domains[domain];
  return updateDomain(settings, domain, {
    mode,
    start: current.start ?? DEFAULT_WINDOW_START,
    end: current.end ?? DEFAULT_WINDOW_END,
  });
}

export function updateDomainTime(
  settings: DetectionSettings,
  domain: DetectionDomainKey,
  field: 'start' | 'end',
  value: string,
): DetectionSettings {
  return updateDomain(settings, domain, { [field]: value });
}

export function validateDetectionSettings(settings: DetectionSettings): string | null {
  for (const domain of DOMAIN_ORDER) {
    const setting = settings.domains[domain];
    if (setting.mode === 'window' && (!setting.start || !setting.end)) {
      return '시간 지정 모드에서는 시작·종료 시간을 모두 입력하세요.';
    }
  }
  return null;
}
