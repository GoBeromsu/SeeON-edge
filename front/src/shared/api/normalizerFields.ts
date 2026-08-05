export function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value);
}

export function pickString(record: Record<string, unknown>, keys: string[], defaultValue = ''): string {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === 'string' && value.trim()) {
      return value;
    }
  }
  return defaultValue;
}

export function pickNullableString(record: Record<string, unknown>, keys: string[]): string | null {
  const value = pickString(record, keys);
  return value || null;
}

export function pickNumber(record: Record<string, unknown>, keys: string[]): number | null {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === 'number' && Number.isFinite(value)) {
      return value;
    }
  }
  return null;
}

export function pickNonNegativeNumber(record: Record<string, unknown>, keys: string[]): number | null {
  const value = pickNumber(record, keys);
  return value !== null && value >= 0 ? value : null;
}

export function pickBoolean(record: Record<string, unknown>, keys: string[]): boolean | null {
  for (const key of keys) {
    const value = record[key];
    if (typeof value === 'boolean') {
      return value;
    }
  }
  return null;
}

export function isNullableString(value: unknown): value is string | null {
  return value === null || typeof value === 'string';
}

export function isNullableBoolean(value: unknown): value is boolean | null {
  return value === null || typeof value === 'boolean';
}

export function isNullableFiniteNumber(value: unknown): value is number | null {
  return value === null || (typeof value === 'number' && Number.isFinite(value));
}

export function isNullableInteger(value: unknown): value is number | null {
  return value === null || (typeof value === 'number' && Number.isInteger(value));
}

export function isNonEmptyString(value: unknown): value is string {
  return typeof value === 'string' && value.trim().length > 0;
}

export function hasNullableString(record: Record<string, unknown>, key: string): boolean {
  return !(key in record) || isNullableString(record[key]);
}

export function hasNullableInteger(record: Record<string, unknown>, key: string): boolean {
  return !(key in record) || isNullableInteger(record[key]);
}

export function isNullableNonNegativeNumber(value: unknown): value is number | null {
  return value === null || (typeof value === 'number' && Number.isFinite(value) && value >= 0);
}

/** True when `key` is absent, or present as a non-negative finite number, or explicitly null. */
export function hasNullableNonNegativeNumber(record: Record<string, unknown>, key: string): boolean {
  return !(key in record) || isNullableNonNegativeNumber(record[key]);
}
