import { isNullableBoolean, isNullableInteger, isNullableString, isRecord } from '@/shared/api/normalizerFields';
import type {
  ConnectionTestResult,
  ConnectionView,
  HeartbeatRelayStatus,
} from '@/shared/api/types';

const CONNECTION_ERRORS = ['unreachable', 'timeout', 'auth', 'conflict', 'invalid_response'] as const;
const HEARTBEAT_ERRORS = ['auth', 'timeout', 'unreachable'] as const;
const DEFAULT_HEARTBEAT: HeartbeatRelayStatus = {
  enabled: false,
  last_success_at: null,
  last_error_class: null,
  detail: null,
};

function isNullableEnum<T extends string>(value: unknown, allowed: readonly T[]): value is T | null {
  return value === null || (typeof value === 'string' && allowed.some((item) => item === value));
}

function normalizeHeartbeat(value: unknown): HeartbeatRelayStatus {
  if (!isRecord(value) || typeof value.enabled !== 'boolean'
    || !('last_success_at' in value) || !isNullableString(value.last_success_at)
    || !('last_error_class' in value) || !isNullableEnum(value.last_error_class, HEARTBEAT_ERRORS)
    || !('detail' in value) || !isNullableString(value.detail)) return DEFAULT_HEARTBEAT;
  return { enabled: value.enabled, last_success_at: value.last_success_at,
    last_error_class: value.last_error_class, detail: value.detail };
}

export function normalizeConnectionView(value: unknown): ConnectionView {
  if (!isRecord(value)
    || !('events_url' in value) || !isNullableString(value.events_url)
    || !('config_url' in value) || !isNullableString(value.config_url)
    || !('facility_code' in value) || !isNullableString(value.facility_code)
    || !('client_installation_ref' in value) || !isNullableString(value.client_installation_ref)
    || !('facility_id' in value) || !isNullableString(value.facility_id)
    || !('edge_installation_id' in value) || !isNullableString(value.edge_installation_id)
    || !('enrollment_generation' in value) || !isNullableInteger(value.enrollment_generation)
    || typeof value.facility_token_set !== 'boolean'
    || !('facility_token_masked' in value) || !isNullableString(value.facility_token_masked)
    || typeof value.enrolled !== 'boolean' || typeof value.configured !== 'boolean'
    || !('reachable' in value) || !isNullableBoolean(value.reachable)
    || !('last_ok_at' in value) || !isNullableString(value.last_ok_at)
    || !('updated_at' in value) || !isNullableString(value.updated_at)) throw new Error('Invalid connection response');
  return {
    events_url: value.events_url,
    config_url: value.config_url,
    facility_code: value.facility_code,
    client_installation_ref: value.client_installation_ref,
    facility_id: value.facility_id,
    edge_installation_id: value.edge_installation_id,
    enrollment_generation: value.enrollment_generation,
    facility_token_set: value.facility_token_set,
    facility_token_masked: value.facility_token_masked,
    enrolled: value.enrolled,
    configured: value.configured,
    reachable: value.reachable,
    last_ok_at: value.last_ok_at,
    updated_at: value.updated_at,
    heartbeat_relay: normalizeHeartbeat(value.heartbeat_relay),
  };
}

export function normalizeConnectionTestResult(value: unknown): ConnectionTestResult {
  if (!isRecord(value) || typeof value.ok !== 'boolean'
    || !('error_class' in value) || !isNullableEnum(value.error_class, CONNECTION_ERRORS)
    || typeof value.detail !== 'string'
    || !('facility_id' in value) || !isNullableString(value.facility_id)
    || !('edge_installation_id' in value) || !isNullableString(value.edge_installation_id)
    || !('enrollment_generation' in value) || !isNullableInteger(value.enrollment_generation)) {
    throw new Error('Invalid connection test response');
  }
  return { ok: value.ok, error_class: value.error_class, detail: value.detail,
    facility_id: value.facility_id, edge_installation_id: value.edge_installation_id,
    enrollment_generation: value.enrollment_generation };
}
