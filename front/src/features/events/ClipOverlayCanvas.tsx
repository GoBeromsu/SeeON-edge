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

/** Same wording as the live view: `사람 84%` and `침대1`; clips carry no fall/bed-exit state. */
export function personLabel(confidence: number): string {
  return `사람 ${Math.round(confidence * 100)}%`;
}

export function bedLabel(index: number): string {
  return `침대${index + 1}`;
}

const LABEL_FONT = '600 12px system-ui, sans-serif';
const LABEL_PADDING = 4;
const LABEL_HEIGHT = 18;

/** Label chip position kept inside the drawable box (like the live preview renderer). */
export function labelOrigin(x: number, y: number, width: number, drawableWidth: number): { left: number; top: number } {
  return {
    left: Math.max(0, Math.min(x, drawableWidth - width)),
    top: y - LABEL_HEIGHT >= 0 ? y - LABEL_HEIGHT : y,
  };
}

function drawLabel(context: CanvasRenderingContext2D, text: string, x: number, y: number, color: string, drawableWidth: number): void {
  context.font = LABEL_FONT;
  const width = context.measureText(text).width + LABEL_PADDING * 2;
  const { left, top } = labelOrigin(x, y, width, drawableWidth);
  context.fillStyle = color;
  context.fillRect(left, top, width, LABEL_HEIGHT);
  context.fillStyle = '#111111';
  context.textBaseline = 'middle';
  context.fillText(text, left + LABEL_PADDING, top + LABEL_HEIGHT / 2);
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
        result.bed_geometries.forEach(({ points }, index) => {
          context.beginPath();
          points.forEach(([x, y], pointIndex) => pointIndex === 0 ? context.moveTo(offsetX + x * scale, offsetY + y * scale) : context.lineTo(offsetX + x * scale, offsetY + y * scale));
          context.closePath();
          context.stroke();
          const top = points.reduce((best, point) => (point[1] < best[1] ? point : best), points[0]);
          drawLabel(context, bedLabel(index), offsetX + top[0] * scale, offsetY + top[1] * scale, '#38bdf8', bounds.width);
        });
      }
      if (personEnabled) {
        context.strokeStyle = '#f97316';
        frame.boxes.forEach(({ x1, y1, x2, y2, confidence }) => {
          context.strokeRect(offsetX + x1 * scale, offsetY + y1 * scale, (x2 - x1) * scale, (y2 - y1) * scale);
          drawLabel(context, personLabel(confidence), offsetX + x1 * scale, offsetY + y1 * scale, '#f97316', bounds.width);
        });
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
