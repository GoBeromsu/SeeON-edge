import { isRecord, pickNonNegativeNumber, pickNullableString, pickString } from '@/shared/api/normalizerFields';
import type { ClipStorageBrowseEntry, ClipStorageBrowseResult, ClipStorageInfo } from '@/shared/api/types';

export function normalizeClipStorageInfo(value: unknown): ClipStorageInfo {
  const record = isRecord(value) ? value : null;
  if (!record || typeof record.root !== 'string' || typeof record.selected_path !== 'string') {
    throw new Error('Invalid clip storage response');
  }

  return {
    root: record.root,
    selected_path: record.selected_path,
    total_bytes: pickNonNegativeNumber(record, ['total_bytes']),
    used_bytes: pickNonNegativeNumber(record, ['used_bytes']),
    used_pct: pickNonNegativeNumber(record, ['used_pct']),
  };
}

function normalizeBrowseEntry(value: unknown): ClipStorageBrowseEntry | null {
  if (!isRecord(value)) return null;
  const name = pickString(value, ['name']);
  const path = pickString(value, ['path']);
  if (!name || !path) return null;
  return { name, path };
}

export function normalizeClipStorageBrowse(value: unknown): ClipStorageBrowseResult {
  const record = isRecord(value) ? value : null;
  if (!record || typeof record.path !== 'string' || !Array.isArray(record.directories)) {
    throw new Error('Invalid clip storage browse response');
  }

  return {
    path: record.path,
    parent: pickNullableString(record, ['parent']),
    directories: record.directories
      .map(normalizeBrowseEntry)
      .filter((entry): entry is ClipStorageBrowseEntry => entry !== null),
  };
}
