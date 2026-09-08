import type { ClipAnalysisBedGeometry, ClipAnalysisBox, ClipAnalysisFrame, ClipAnalysisResult, ClipAnalysisStatus } from '@/shared/api/types';
import { isRecord } from '@/shared/api/normalizerFields';

function rejectUnknown(record: Record<string, unknown>, allowed: readonly string[]): void {
  if (Object.keys(record).some((key) => !allowed.includes(key))) throw new Error('Unexpected clip analysis field');
}

function number(value: unknown): number {
  if (typeof value !== 'number' || !Number.isFinite(value)) throw new Error('Invalid clip analysis number');
  return value;
}

function positiveInteger(value: unknown): number {
  const parsed = number(value);
  if (!Number.isInteger(parsed) || parsed <= 0) throw new Error('Invalid clip analysis positive integer');
  return parsed;
}

function box(value: unknown): ClipAnalysisBox {
  if (!isRecord(value)) throw new Error('Invalid clip analysis box');
  rejectUnknown(value, ['x1', 'y1', 'x2', 'y2', 'confidence']);
  return { x1: number(value.x1), y1: number(value.y1), x2: number(value.x2), y2: number(value.y2), confidence: number(value.confidence) };
}

function frame(value: unknown): ClipAnalysisFrame {
  if (!isRecord(value)) throw new Error('Invalid clip analysis frame');
  rejectUnknown(value, ['pts', 'status', 'boxes']);
  if (value.status !== 'available' && value.status !== 'no_evidence' && value.status !== 'ambiguous_timestamp') throw new Error('Invalid clip analysis frame status');
  if (!Array.isArray(value.boxes)) throw new Error('Invalid clip analysis boxes');
  return { pts: number(value.pts), status: value.status, boxes: value.boxes.map(box) };
}

function geometry(value: unknown): ClipAnalysisBedGeometry {
  if (!isRecord(value)) throw new Error('Invalid clip analysis bed geometry');
  rejectUnknown(value, ['points', 'provenance_pts']);
  if (!Array.isArray(value.points) || value.points.length < 3) throw new Error('Invalid clip analysis bed points');
  const points = value.points.map((point): [number, number] => {
    if (!Array.isArray(point) || point.length !== 2) throw new Error('Invalid clip analysis bed point');
    return [number(point[0]), number(point[1])];
  });
  return { points, provenance_pts: number(value.provenance_pts) };
}

function result(value: unknown): ClipAnalysisResult {
  if (!isRecord(value)) throw new Error('Invalid clip analysis result');
  rejectUnknown(value, [
    'source',
    'clip_id',
    'clip_sha256',
    'pose_model_sha256',
    'bed_model_sha256',
    'decoder_identity',
    'analysis_profile_sha256',
    'time_base',
    'frames',
    'bed_geometries',
    'image_width',
    'image_height',
  ]);
  if (value.source !== 'clip_reanalysis') throw new Error('Invalid clip analysis source');
  for (const key of ['clip_sha256', 'pose_model_sha256', 'bed_model_sha256', 'analysis_profile_sha256'] as const) {
    if (typeof value[key] !== 'string' || !/^[0-9a-f]{64}$/.test(value[key])) throw new Error(`Invalid clip analysis ${key}`);
  }
  if (typeof value.clip_id !== 'string' || value.clip_id.length === 0) throw new Error('Invalid clip analysis clip_id');
  if (typeof value.decoder_identity !== 'string' || value.decoder_identity.length === 0) throw new Error('Invalid clip analysis decoder_identity');
  if (!isRecord(value.time_base)) throw new Error('Invalid clip analysis time base');
  rejectUnknown(value.time_base, ['numerator', 'denominator']);
  if (!Array.isArray(value.frames) || !Array.isArray(value.bed_geometries)) throw new Error('Invalid clip analysis collections');
  return {
    time_base: { numerator: positiveInteger(value.time_base.numerator), denominator: positiveInteger(value.time_base.denominator) },
    frames: value.frames.map(frame),
    bed_geometries: value.bed_geometries.map(geometry),
    image_width: positiveInteger(value.image_width),
    image_height: positiveInteger(value.image_height),
  };
}

export function normalizeClipAnalysisStatus(value: unknown): ClipAnalysisStatus {
  if (!isRecord(value)) throw new Error('Invalid clip analysis status');
  rejectUnknown(value, ['state', 'reason', 'served_media_sha256', 'served_timing_identical', 'result']);
  if (value.state !== 'idle' && value.state !== 'queued' && value.state !== 'running' && value.state !== 'available' && value.state !== 'failed' && value.state !== 'unavailable') throw new Error('Invalid clip analysis state');
  if ('reason' in value && typeof value.reason !== 'string') throw new Error('Invalid clip analysis reason');
  if (typeof value.served_media_sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(value.served_media_sha256)) throw new Error('Invalid served media digest');
  if (value.state === 'available') {
    if (value.served_timing_identical !== true || !('result' in value)) throw new Error('Incomplete available clip analysis');
    return { state: value.state, served_media_sha256: value.served_media_sha256, served_timing_identical: value.served_timing_identical, result: result(value.result) };
  }
  if ('served_timing_identical' in value || 'result' in value) throw new Error('Unexpected nonavailable clip analysis detail');
  if (typeof value.reason === 'string') return { state: value.state, reason: value.reason, served_media_sha256: value.served_media_sha256 };
  return { state: value.state, served_media_sha256: value.served_media_sha256 };
}
