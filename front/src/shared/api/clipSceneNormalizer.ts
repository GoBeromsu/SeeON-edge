import { isRecord } from '@/shared/api/normalizerFields';
import type { ClipScene, ClipSceneFrame } from '@/shared/api/types';

const SCENE_INDEX_SCHEMA_VERSION = 1;

export function normalizeClipScene(value: unknown): ClipScene | null {
  if (!isRecord(value) || !hasOnlyKeys(value, TOP_LEVEL_KEYS)
    || !isNonEmptyString(value.camera_id) || !isNonEmptyString(value.clip_id)
    || value.coordinate_space !== 'source-pixels'
    || !isNonNegativeInteger(value.scene_index_schema_version)) {
    throw new Error('Invalid clip scene response');
  }
  if (value.scene_index_schema_version > SCENE_INDEX_SCHEMA_VERSION) return null;
  if (!isNonNegativeInteger(value.scene_schema_version)
    || !isNonNegativeInteger(value.detail_shed_frame_count)
    || !isNonNegativeInteger(value.frame_count)
    || !isPair(value.source_dimensions) || value.source_dimensions.some((item) => item <= 0)
    || !Array.isArray(value.components) || !value.components.every(isComponent)
    || !Array.isArray(value.decision_provenance) || !value.decision_provenance.every(isDecisionProvenance)
    || !Array.isArray(value.frames) || value.frames.length !== value.frame_count
    || !value.frames.every(isFrame)
    || !isStreamIdentity(value.stream_identity) || !isStyle(value.style) || !isTimeOrigin(value.time_origin)) {
    throw new Error('Invalid clip scene response');
  }
  return value as ClipScene;
}

const TOP_LEVEL_KEYS = new Set([
  'camera_id', 'clip_id', 'components', 'coordinate_space', 'decision_provenance',
  'detail_shed_frame_count', 'frame_count', 'frames', 'scene_index_schema_version',
  'scene_schema_version', 'source_dimensions', 'stream_identity', 'style', 'time_origin',
]);
const FRAME_KEYS = new Set(['p', 'q', 'sd', 't', 'bd', 'dc', 'lb', 'ps']);
const PERSON_KEYS = new Set(['b', 'c', 'i', 'tr', 'tr_r', 'k']);
const BED_KEYS = new Set(['b', 'c', 'ct', 'i', 'pg', 'pv', 'sm']);
const LABEL_KEYS = new Set(['x', 'y', 't', 'c', 'z']);
const CONTAINMENT_KEYS = new Set(['r', 's', 'th', 'tr']);
const DECISION_KEYS = new Set(['bd', 'm', 'p', 'ps', 'rs', 'rm', 's', 'sc', 'th', 'tg', 'tr', 'e', 'cn', 'cn_t']);

function isFrame(value: unknown): value is ClipSceneFrame {
  return isRecord(value) && hasOnlyKeys(value, FRAME_KEYS)
    && isFiniteNumber(value.p) && isNonNegativeInteger(value.q) && typeof value.sd === 'boolean'
    && isFiniteNumber(value.t) && Array.isArray(value.bd) && value.bd.every(isBed)
    && Array.isArray(value.dc) && value.dc.every(isDecision) && Array.isArray(value.lb) && value.lb.every(isLabel)
    && Array.isArray(value.ps) && value.ps.every(isPerson);
}

function isPerson(value: unknown): boolean {
  return isRecord(value) && hasOnlyKeys(value, PERSON_KEYS)
    && isBox(value.b) && isFiniteNumber(value.c) && isNonNegativeInteger(value.i)
    && (typeof value.tr === 'number' || value.tr === null)
    && (value.tr_r === undefined || typeof value.tr_r === 'string')
    && (value.k === undefined || (Array.isArray(value.k) && value.k.every(isKeypoint)));
}

function isBed(value: unknown): boolean {
  return isRecord(value) && hasOnlyKeys(value, BED_KEYS)
    && isBox(value.b) && isFiniteNumber(value.c) && isNonNegativeInteger(value.i)
    && Array.isArray(value.ct) && value.ct.every(isContainment) && Array.isArray(value.pg) && value.pg.every(isPair)
    && typeof value.pv === 'string' && typeof value.sm === 'string';
}

function isLabel(value: unknown): boolean {
  return isRecord(value) && hasOnlyKeys(value, LABEL_KEYS)
    && isFiniteNumber(value.x) && isFiniteNumber(value.y) && typeof value.t === 'string'
    && isColor(value.c) && isFiniteNumber(value.z);
}

function isContainment(value: unknown): boolean {
  return isRecord(value) && hasOnlyKeys(value, CONTAINMENT_KEYS)
    && isNullableFiniteNumber(value.r) && typeof value.s === 'string'
    && isNullableFiniteNumber(value.th) && isNullableFiniteNumber(value.tr);
}

function isDecision(value: unknown): boolean {
  return isRecord(value) && hasOnlyKeys(value, DECISION_KEYS)
    && isNullableFiniteNumber(value.bd) && isNonEmptyString(value.m) && isNonEmptyString(value.p)
    && typeof value.ps === 'string' && typeof value.rs === 'string' && isNonEmptyString(value.rm)
    && typeof value.s === 'string' && isNullableFiniteNumber(value.sc) && isNullableFiniteNumber(value.th)
    && typeof value.tg === 'boolean' && isNullableFiniteNumber(value.tr) && isNonEmptyString(value.e)
    && Array.isArray(value.cn) && value.cn.every((counter) => Array.isArray(counter) && counter.length === 2 && isNonEmptyString(counter[0]) && isNullableFiniteNumber(counter[1]))
    && (value.cn_t === undefined || typeof value.cn_t === 'boolean');
}

function isKeypoint(value: unknown): boolean {
  return Array.isArray(value) && value.length === 4 && isNonNegativeInteger(value[0])
    && (value[1] === null || isFiniteNumber(value[1])) && (value[2] === null || isFiniteNumber(value[2]))
    && isFiniteNumber(value[3]);
}

function isComponent(value: unknown): boolean {
  return isRecord(value) && hasOnlyKeys(value, new Set(['id', 'sm']))
    && isNonEmptyString(value.id) && typeof value.sm === 'string';
}

function isDecisionProvenance(value: unknown): boolean {
  return isRecord(value) && hasOnlyKeys(value, new Set(['m', 'e', 'p', 'rm']))
    && isNonEmptyString(value.m) && isNonEmptyString(value.e)
    && isNonEmptyString(value.p) && isNonEmptyString(value.rm);
}

function isStreamIdentity(value: unknown): boolean {
  return isRecord(value) && hasOnlyKeys(value, new Set(['generation', 'stream_epoch', 'worker_boot_id']))
    && isNonNegativeInteger(value.generation) && isNonNegativeInteger(value.stream_epoch)
    && isNonEmptyString(value.worker_boot_id);
}

function isStyle(value: unknown): boolean {
  if (!isRecord(value) || !hasOnlyKeys(value, new Set(['palette', 'skeleton', 'z_order']))
    || !isRecord(value.palette) || !hasOnlyKeys(value.palette, new Set(['bed', 'danger', 'neutral', 'person', 'pose', 'pose_dot']))
    || !Object.values(value.palette).every(isColor) || !isRecord(value.skeleton)
    || !hasOnlyKeys(value.skeleton, new Set(['edges'])) || !Array.isArray(value.skeleton.edges)
    || !value.skeleton.edges.every(isPair) || !isRecord(value.z_order)
    || !hasOnlyKeys(value.z_order, new Set(['bed', 'decision', 'person']))) return false;
  return Object.values(value.z_order).every(isFiniteNumber);
}

function isTimeOrigin(value: unknown): boolean {
  return isRecord(value) && hasOnlyKeys(value, new Set(['event_pts_sec', 'media_origin_pts_sec', 'requested_end_pts_sec', 'requested_start_pts_sec']))
    && Object.values(value).every(isFiniteNumber);
}

function hasOnlyKeys(value: Record<string, unknown>, keys: Set<string>): boolean {
  return Object.keys(value).every((key) => keys.has(key));
}
function isNonEmptyString(value: unknown): value is string { return typeof value === 'string' && value.length > 0; }
function isFiniteNumber(value: unknown): value is number { return typeof value === 'number' && Number.isFinite(value); }
function isNullableFiniteNumber(value: unknown): boolean { return value === null || isFiniteNumber(value); }
function isNonNegativeInteger(value: unknown): value is number { return Number.isInteger(value) && typeof value === 'number' && value >= 0; }
function isPair(value: unknown): value is [number, number] { return Array.isArray(value) && value.length === 2 && value.every(isFiniteNumber); }
function isBox(value: unknown): value is [number, number, number, number] { return Array.isArray(value) && value.length === 4 && value.every(isFiniteNumber); }
function isColor(value: unknown): value is [number, number, number] { return Array.isArray(value) && value.length === 3 && value.every(isFiniteNumber); }
