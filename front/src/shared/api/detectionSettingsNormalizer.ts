import { isRecord, pickNullableString } from '@/shared/api/normalizerFields';
import type { DetectionDomainKey, DetectionDomainSetting, DetectionSettings } from '@/shared/api/types';

const DOMAIN_KEYS: readonly DetectionDomainKey[] = ['fall', 'bed_exit'];

function normalizeDomainSetting(value: unknown): DetectionDomainSetting {
  const record = isRecord(value) ? value : {};
  return {
    on: record.on === true,
    mode: record.mode === 'window' ? 'window' : 'always',
    start: pickNullableString(record, ['start']),
    end: pickNullableString(record, ['end']),
  };
}

/** A domain missing from the response (older/partial backend) normalizes to an all-off default rather than throwing. */
export function normalizeDetectionSettings(value: unknown): DetectionSettings {
  const record = isRecord(value) ? value : null;
  const domains = record && isRecord(record.domains) ? record.domains : null;
  if (!record || !domains) {
    throw new Error('Invalid detection settings response');
  }

  const normalizedDomains = {} as DetectionSettings['domains'];
  for (const key of DOMAIN_KEYS) {
    normalizedDomains[key] = normalizeDomainSetting(domains[key]);
  }
  return { domains: normalizedDomains };
}
