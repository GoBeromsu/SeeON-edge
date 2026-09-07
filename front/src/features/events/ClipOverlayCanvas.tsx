import { useEffect, useRef } from 'react';
import type { ClipAnalysisFrame, ClipAnalysisResult } from '@/shared/api/types';

const MATCH_TOLERANCE_SECONDS = 0.0005;

type ContainTransform = { scale: number; offsetX: number; offsetY: number };

export function findMatchingFrame(result: ClipAnalysisResult, mediaTime: number): ClipAnalysisFrame | null {
  const matches = result.frames.filter((frame) => Math.abs(frame.pts * result.time_base.numerator / result.time_base.denominator - mediaTime) <= MATCH_TOLERANCE_SECONDS);
  return matches.length === 1 && matches[0]?.status === 'available' ? matches[0] : null;
}

export function containTransform(width: number, height: number, imageWidth: number, imageHeight: number): ContainTransform {
  const scale = Math.min(width / imageWidth, height / imageHeight);
  return { scale, offsetX: (width - imageWidth * scale) / 2, offsetY: (height - imageHeight * scale) / 2 };
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
  const lastFrameRef = useRef<ClipAnalysisFrame | null>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || !video || !result || !('requestVideoFrameCallback' in video)) return undefined;
    const draw = (frame: ClipAnalysisFrame | null): void => {
      const bounds = video.getBoundingClientRect();
      const context = canvas.getContext('2d');
      if (!context || bounds.width === 0 || bounds.height === 0) return;
      const ratio = devicePixelRatio;
      canvas.width = Math.round(bounds.width * ratio);
      canvas.height = Math.round(bounds.height * ratio);
      canvas.style.width = `${bounds.width}px`;
      canvas.style.height = `${bounds.height}px`;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, bounds.width, bounds.height);
      if (!frame) return;
      const { scale, offsetX, offsetY } = containTransform(bounds.width, bounds.height, result.image_width, result.image_height);
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
      if (personEnabled) {
        context.strokeStyle = '#f97316';
        frame.boxes.forEach(({ x1, y1, x2, y2 }) => context.strokeRect(offsetX + x1 * scale, offsetY + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale));
      }
    };
    const render = (mediaTime: number): void => {
      const frame = findMatchingFrame(result, mediaTime);
      lastFrameRef.current = frame;
      draw(frame);
    };
    const schedule = (): void => {
      callbackRef.current = video.requestVideoFrameCallback((_now, metadata) => {
        render(metadata.mediaTime);
        schedule();
      });
    };
    const clear = (): void => {
      lastFrameRef.current = null;
      draw(null);
    };
    const redraw = (): void => draw(lastFrameRef.current);
    const observer = new ResizeObserver(redraw);
    observer.observe(video);
    video.addEventListener('seeking', clear);
    video.addEventListener('pause', redraw);
    redraw();
    schedule();
    return () => {
      if (callbackRef.current !== null) video.cancelVideoFrameCallback(callbackRef.current);
      observer.disconnect();
      video.removeEventListener('seeking', clear);
      video.removeEventListener('pause', redraw);
    };
  }, [bedEnabled, personEnabled, result, video]);

  return result && video && 'requestVideoFrameCallback' in video
    ? <canvas ref={canvasRef} aria-hidden="true" className="pointer-events-none absolute inset-0" />
    : null;
}
