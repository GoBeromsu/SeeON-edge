import { describe, expect, it } from 'vitest';
import {
  DOMAIN_LABELS,
  DOMAIN_ORDER,
  formatDomainSchedule,
  updateDomainMode,
  updateDomainOn,
  updateDomainTime,
  validateDetectionSettings,
} from '@/features/settings/detectionSettingsForm';
import type { DetectionSettings } from '@/shared/api/types';

const baseSettings: DetectionSettings = {
  domains: {
    fall: { on: true, mode: 'always', start: null, end: null },
    bed_exit: { on: true, mode: 'window', start: '21:00', end: '06:00' },
  },
};

describe('detectionSettingsForm', () => {
  it('lists 침대 이탈 before 낙상, matching the README read-mode example order', () => {
    expect(DOMAIN_ORDER).toEqual(['bed_exit', 'fall']);
    expect(DOMAIN_LABELS.bed_exit).toBe('침대 이탈');
    expect(DOMAIN_LABELS.fall).toBe('낙상');
  });

  it('formats an "always" domain as 항상 and a "window" domain as its start–end range', () => {
    expect(formatDomainSchedule(baseSettings.domains.fall)).toBe('항상');
    expect(formatDomainSchedule(baseSettings.domains.bed_exit)).toBe('21:00–06:00');
  });

  it('toggles a domain on/off without touching the other domain', () => {
    const next = updateDomainOn(baseSettings, 'fall', false);
    expect(next.domains.fall.on).toBe(false);
    expect(next.domains.bed_exit).toBe(baseSettings.domains.bed_exit);
  });

  it('clears start/end when switching a domain to "always"', () => {
    const next = updateDomainMode(baseSettings, 'bed_exit', 'always');
    expect(next.domains.bed_exit).toEqual({ on: true, mode: 'always', start: null, end: null });
  });

  it('fills a default night-shift window when switching a domain to "window" with no existing time', () => {
    const next = updateDomainMode(baseSettings, 'fall', 'window');
    expect(next.domains.fall).toEqual({ on: true, mode: 'window', start: '21:00', end: '06:00' });
  });

  it('preserves an existing window when re-selecting "window" mode', () => {
    const withCustomWindow: DetectionSettings = {
      domains: { ...baseSettings.domains, fall: { on: true, mode: 'window', start: '22:00', end: '05:00' } },
    };
    const next = updateDomainMode(withCustomWindow, 'fall', 'window');
    expect(next.domains.fall).toEqual({ on: true, mode: 'window', start: '22:00', end: '05:00' });
  });

  it('updates a single time field for a domain', () => {
    const next = updateDomainTime(baseSettings, 'bed_exit', 'start', '22:30');
    expect(next.domains.bed_exit.start).toBe('22:30');
    expect(next.domains.bed_exit.end).toBe('06:00');
  });

  it('rejects a window-mode domain missing either time', () => {
    const missingStart: DetectionSettings = {
      domains: { ...baseSettings.domains, bed_exit: { on: true, mode: 'window', start: null, end: '06:00' } },
    };
    expect(validateDetectionSettings(missingStart)).toMatch(/시작·종료 시간을 모두 입력/);
  });

  it('accepts settings where every window-mode domain has both times, and every always-mode domain needs none', () => {
    expect(validateDetectionSettings(baseSettings)).toBeNull();
  });
});
