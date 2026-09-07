import { useEffect, useRef } from 'react';
import type { ClipAnalysisFrame, ClipAnalysisResult } from '@/shared/api/types';

const MATCH_TOLERANCE_SECONDS = 0.0005;

export function findMatchingFrame(result: ClipAnalysisResult, mediaTime: number): ClipAnalysisFrame | null {
  const matches = result.frames.filter((frame) => Math.abs((frame.pts * result.time_base.numerator / result.time_base.denominator) - mediaTime) <= MATCH_TOLERANCE_SECONDS);
  return matches.length === 1 ? matches[0] ?? null : null;
}

type Props = {
  video: HTMLVideoElement | null;
  result: ClipAnalysisResult | undefined;
  personEnabled: boolean;
  bedEnabled: boolean;
};

export function ClipOverlayCanvas({ video, result, personEnabled, bedEnabled }: Props): JSX.Element | null {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const callbackRef = useRef<number | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !video || !result || !('requestVideoFrameCallback' in video)) return undefined;
    const render = (mediaTime: number): void => {
      const bounds = video.getBoundingClientRect();
      const context = canvas.getContext('2d');
      if (!context || bounds.width === 0 || bounds.height === 0) return;
      canvas.width = Math.round(bounds.width * devicePixelRatio);
      canvas.height = Math.round(bounds.height * devicePixelRatio);
      canvas.style.width = `${bounds.width}px`;
      canvas.style.height = `${bounds.height}px`;
      context.scale(devicePixelRatio, devicePixelRatio);
      context.clearRect(0, 0, bounds.width, bounds.height);
      const scale = Math.min(bounds.width / result.image_width, bounds.height / result.image_height);
      const offsetX = (bounds.width - result.image_width * scale) / 2;
      const offsetY = (bounds.height - result.image_height * scale) / 2;
      context.lineWidth = 2;
      if (bedEnabled) {
        context.strokeStyle = '#38bdf8';
        result.bed_geometries.forEach(({ points }) => {
          context.beginPath();
          points.forEach(([x, y], index) => index === 0 ? context.moveTo(offsetX + x * scale, offsetY + y * scale) : context.lineTo(offsetX + x * scale, offsetY + y * scale));
          context.closePath();
          context.stroke();
        });
      }
      const frame = findMatchingFrame(result, mediaTime);
      if (personEnabled && frame?.status === 'available') {
        context.strokeStyle = '#f97316';
        frame.boxes.forEach(({ x1, y1, x2, y2 }) => context.strokeRect(offsetX + x1 * scale, offsetY + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale));
      }
    };
    const schedule = (): void => {
      callbackRef.current = video.requestVideoFrameCallback((_now, metadata) => {
        render(metadata.mediaTime);
        schedule();
      });
    };
    const clear = (): void => {
      const context = canvas.getContext('2d');
      context?.clearRect(0, 0, canvas.width, canvas.height);
    };
    video.addEventListener('seeking', clear);
    schedule();
    return () => {
      if (callbackRef.current !== null) video.cancelVideoFrameCallback(callbackRef.current);
      video.removeEventListener('seeking', clear);
    };
  }, [bedEnabled, personEnabled, result, video]);

  return result && video && 'requestVideoFrameCallback' in video
    ? <canvas ref={canvasRef} aria-hidden="true" className="pointer-events-none absolute inset-0" />
    : null;
}
