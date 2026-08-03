export function formatClipTimestamp(value: string | null): string {
  if (!value) return '시간 정보 없음';
  const date = new Date(value);
  if (!Number.isFinite(date.getTime())) return '시간 정보 없음';
  return date.toLocaleString('ko-KR', {
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
  });
}

export function formatDuration(seconds: number | null): string {
  if (seconds === null || !Number.isFinite(seconds) || seconds < 0) return '확인 중…';
  const total = Math.round(seconds);
  const minutes = Math.floor(total / 60);
  const secs = total % 60;
  return `${minutes}:${secs.toString().padStart(2, '0')}`;
}

export function formatResolution(width: number | null, height: number | null): string {
  if (!width || !height) return '확인 중…';
  return `${width}×${height}`;
}
