import { isNullableBoolean, isNullableString, isRecord } from '@/shared/api/normalizerFields';
import type {
  ConnectionErrorClass,
  ConnectionTestResult,
  ConnectionView,
  HeartbeatRelayErrorClass,
  HeartbeatRelayStatus,
} from '@/shared/api/types';

const CONNECTION_ERROR_CLASSES: readonly ConnectionErrorClass[] = [
  'unconfigured',
  'invalid_url',
  'unreachable',
  'timeout',
  'auth',
];

const HEARTBEAT_RELAY_ERROR_CLASSES: readonly HeartbeatRelayErrorClass[] = ['auth', 'timeout', 'unreachable'];

const DEFAULT_HEARTBEAT_RELAY: HeartbeatRelayStatus = {
  enabled: false,
  last_success_at: null,
  last_error_class: null,
  detail: null,
};

function isNullableEnum<T extends string>(value: unknown, allowed: readonly T[]): value is T | null {
  return value === null || (typeof value === 'string' && (allowed as readonly string[]).includes(value));
}

/**
 * Unlike the rest of this file, malformed/missing input here never throws: an older backend simply
 * hasn't shipped this field yet, so it normalizes to the all-disabled default and the rest of the
 * connection view still loads.
 */
function normalizeHeartbeatRelay(value: unknown): HeartbeatRelayStatus {
  if (
    !isRecord(value)
    || typeof value.enabled !== 'boolean'
    || !('last_success_at' in value) || !isNullableString(value.last_success_at)
    || !('last_error_class' in value) || !isNullableEnum(value.last_error_class, HEARTBEAT_RELAY_ERROR_CLASSES)
    || !('detail' in value) || !isNullableString(value.detail)
  ) {
    return DEFAULT_HEARTBEAT_RELAY;
  }

  return {
    enabled: value.enabled,
    last_success_at: value.last_success_at,
    last_error_class: value.last_error_class,
    detail: value.detail,
  };
}

export function normalizeConnectionView(value: unknown): ConnectionView {
  const record = isRecord(value) ? value : null;
  if (
    !record
    || !('events_url' in record) || !isNullableString(record.events_url)
    || !('config_url' in record) || !isNullableString(record.config_url)
    || !('facility_id' in record) || !isNullableString(record.facility_id)
    || typeof record.facility_token_set !== 'boolean'
    || !('facility_token_masked' in record) || !isNullableString(record.facility_token_masked)
    || typeof record.configured !== 'boolean'
    || !('reachable' in record) || !isNullableBoolean(record.reachable)
    || !('last_ok_at' in record) || !isNullableString(record.last_ok_at)
    || !('updated_at' in record) || !isNullableString(record.updated_at)
  ) {
    throw new Error('Invalid connection response');
  }

  return {
    events_url: record.events_url,
    config_url: record.config_url,
    facility_id: record.facility_id,
    facility_token_set: record.facility_token_set,
    facility_token_masked: record.facility_token_masked,
    configured: record.configured,
    reachable: record.reachable,
    last_ok_at: record.last_ok_at,
    updated_at: record.updated_at,
    heartbeat_relay: normalizeHeartbeatRelay(record.heartbeat_relay),
  };
}

export function normalizeConnectionTestResult(value: unknown): ConnectionTestResult {
  const record = isRecord(value) ? value : null;
  if (
    !record
    || typeof record.ok !== 'boolean'
    || !('error_class' in record) || !isNullableEnum(record.error_class, CONNECTION_ERROR_CLASSES)
    || typeof record.detail !== 'string'
    || !('probed_url' in record) || !isNullableString(record.probed_url)
  ) {
    throw new Error('Invalid connection test response');
  }

  return {
    ok: record.ok,
    error_class: record.error_class,
    detail: record.detail,
    probed_url: record.probed_url,
  };
}
