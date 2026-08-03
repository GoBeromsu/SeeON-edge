const BYTES_PER_KB = 1000;
const BYTES_PER_MB = BYTES_PER_KB ** 2;
const BYTES_PER_GB = BYTES_PER_KB ** 3;

/** Renders a rounded-to-one-decimal value without a trailing ".0" (e.g. 8.4, but 850 not 850.0). */
function trimDecimal(value: number): string {
  const rounded = Math.round(value * 10) / 10;
  return Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
}

/**
 * Human-readable byte size using decimal (SI, base-1000) units — matches the design spec's example
 * (front/design-handoff/README.md §5 모달 3: an 8,400,000-byte clip shows as "8.4 MB", which only
 * comes out exact under base-1000 division). MB/GB show up to one decimal place, KB/B are whole
 * numbers. Shared by the clip modals (events/operations) and available for reuse elsewhere.
 */
export function formatBytes(bytes: number): string {
  if (!Number.isFinite(bytes) || bytes < 0) return '-';
  if (bytes >= BYTES_PER_GB) return `${trimDecimal(bytes / BYTES_PER_GB)} GB`;
  if (bytes >= BYTES_PER_MB) return `${trimDecimal(bytes / BYTES_PER_MB)} MB`;
  if (bytes >= BYTES_PER_KB) return `${Math.round(bytes / BYTES_PER_KB)} KB`;
  return `${Math.round(bytes)} B`;
}
