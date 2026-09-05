import { useEffect, useId, useRef, useState } from 'react';
import type { BedRegion, BedZonePoint } from '@/shared/api/client';
import { generateUuidV4 } from '@/shared/format/uuid';

type BedZonePolygonEditorProps = {
  regions: readonly BedRegion[];
  imageWidth: number;
  imageHeight: number;
  onChange: (nextRegions: BedRegion[]) => void;
  onDraftValidityChange?: (valid: boolean) => void;
  disabled?: boolean;
};

type Drag = { regionId: string; vertexIndex: number; pointerId: number };

const MAX_REGIONS = 8;
const MAX_VERTICES = 16;

function validPolygon(points: readonly BedZonePoint[]): boolean {
  if (points.length < 3 || new Set(points.map(([x, y]) => `${x},${y}`)).size < 3) return false;
  let twiceArea = 0;
  for (let index = 0; index < points.length; index += 1) {
    const [x1, y1] = points[index];
    const [x2, y2] = points[(index + 1) % points.length];
    twiceArea += x1 * y2 - x2 * y1;
  }
  return Math.abs(twiceArea) > 0.001;
}

function pointsAttribute(points: readonly BedZonePoint[]): string {
  return points.map(([x, y]) => `${x},${y}`).join(' ');
}

function ToolButton({ label, disabled, onClick, children }: {
  label: string;
  disabled?: boolean;
  onClick: () => void;
  children: JSX.Element;
}): JSX.Element {
  return (
    <button type="button" className="icon-button" aria-label={label} title={label} disabled={disabled} onClick={onClick}>
      {children}
    </button>
  );
}

function PrimaryToolButton({ label, disabled, onClick }: {
  label: string;
  disabled?: boolean;
  onClick: () => void;
}): JSX.Element {
  return (
    <button type="button" className="dialog-secondary-action" aria-label={label} title={label} disabled={disabled} onClick={onClick}>
      {label}
    </button>
  );
}

function Icon({ kind }: { kind: 'delete' | 'undo' }): JSX.Element {
  return (
    <svg aria-hidden="true" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.8} strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4">
      {kind === 'delete' ? <><path d="M4 7h16M9 7V4h6v3M7 7l1 13h8l1-13M10 11v5M14 11v5" /></> : null}
      {kind === 'undo' ? <><path d="m9 7-5 5 5 5" /><path d="M20 17a7 7 0 0 0-7-7H4" /></> : null}
    </svg>
  );
}

export function BedZonePolygonEditor({ regions, imageWidth, imageHeight, onChange, onDraftValidityChange, disabled = false }: BedZonePolygonEditorProps): JSX.Element {
  const [draft, setDraft] = useState<{ id: string; points: BedZonePoint[] } | null>(null);
  const [keyboardCursor, setKeyboardCursor] = useState<BedZonePoint>([imageWidth / 2, imageHeight / 2]);
  const [selected, setSelected] = useState<{ regionId: string; vertexIndex: number | null } | null>(null);
  const dragRef = useRef<Drag | null>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const suppressClickRef = useRef(false);
  const instructionsId = useId();

  useEffect(() => {
    setDraft(null);
    setKeyboardCursor([imageWidth / 2, imageHeight / 2]);
    setSelected(null);
    dragRef.current = null;
  }, [imageWidth, imageHeight]);

  useEffect(() => {
    onDraftValidityChange?.(draft === null && regions.every((region) => validPolygon(region.polygon)));
  }, [draft, onDraftValidityChange, regions]);

  function eventPoint(clientX: number, clientY: number): BedZonePoint {
    const rect = svgRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0 || rect.height <= 0) return [0, 0];
    const x = Math.min(imageWidth, Math.max(0, ((clientX - rect.left) / rect.width) * imageWidth));
    const y = Math.min(imageHeight, Math.max(0, ((clientY - rect.top) / rect.height) * imageHeight));
    return [x, y];
  }

  function replaceVertex(regionId: string, vertexIndex: number, point: BedZonePoint): void {
    onChange(regions.map((region) => region.id === regionId
      ? { ...region, origin: 'manual', polygon: region.polygon.map((current, index) => index === vertexIndex ? point : current) }
      : { ...region, polygon: [...region.polygon] }));
  }

  function closeDraft(): void {
    if (!draft || !validPolygon(draft.points)) return;
    onChange([...regions.map((region) => ({ ...region, polygon: [...region.polygon] })), {
      id: draft.id,
      polygon: draft.points,
      origin: 'manual',
    }]);
    setSelected({ regionId: draft.id, vertexIndex: null });
    setDraft(null);
  }

  const selectedRegion = selected ? regions.find((region) => region.id === selected.regionId) : undefined;
  const selectedVertexIndex = selected?.vertexIndex;
  const selectedPoint = selectedRegion && selectedVertexIndex !== null && selectedVertexIndex !== undefined
    ? selectedRegion.polygon[selectedVertexIndex]
    : undefined;

  return (
    <div className="absolute inset-0" data-testid="bed-zone-polygon-editor">
      <svg
        ref={svgRef}
        aria-label="침대 영역 편집 캔버스"
        aria-describedby={instructionsId}
        tabIndex={disabled ? -1 : 0}
        className="absolute inset-0 h-full w-full touch-none"
        viewBox={`0 0 ${imageWidth} ${imageHeight}`}
        preserveAspectRatio="none"
        onKeyDown={(event) => {
          if (event.key === 'Escape' && draft) {
            event.preventDefault();
            setDraft(null);
            return;
          }
          if (!draft || disabled) return;
          const step = event.shiftKey ? 10 : 1;
          if (event.key === 'ArrowLeft' || event.key === 'ArrowRight' || event.key === 'ArrowUp' || event.key === 'ArrowDown') {
            event.preventDefault();
            const [x, y] = keyboardCursor;
            setKeyboardCursor([
              Math.min(imageWidth, Math.max(0, x + (event.key === 'ArrowLeft' ? -step : event.key === 'ArrowRight' ? step : 0))),
              Math.min(imageHeight, Math.max(0, y + (event.key === 'ArrowUp' ? -step : event.key === 'ArrowDown' ? step : 0))),
            ]);
            return;
          }
          if ((event.key === 'Enter' || event.key === ' ') && draft.points.length < MAX_VERTICES) {
            event.preventDefault();
            setDraft({ ...draft, points: [...draft.points, keyboardCursor] });
          }
        }}
        onClick={(event) => {
          if (disabled || !draft || suppressClickRef.current || event.target !== event.currentTarget || draft.points.length >= MAX_VERTICES) {
            suppressClickRef.current = false;
            return;
          }
          setDraft({ ...draft, points: [...draft.points, eventPoint(event.clientX, event.clientY)] });
        }}
        onDoubleClick={(event) => {
          event.preventDefault();
          closeDraft();
        }}
        onPointerMove={(event) => {
          const drag = dragRef.current;
          if (!drag || event.pointerId !== drag.pointerId) return;
          suppressClickRef.current = true;
          replaceVertex(drag.regionId, drag.vertexIndex, eventPoint(event.clientX, event.clientY));
        }}
        onPointerUp={(event) => {
          if (dragRef.current?.pointerId === event.pointerId) dragRef.current = null;
        }}
        onPointerCancel={() => { dragRef.current = null; }}
      >
        {regions.map((region) => (
          <g key={region.id} data-region-id={region.id}>
            <polygon
              points={pointsAttribute(region.polygon)}
              fill="rgba(43,182,163,0.25)"
              stroke="var(--overlay-teal)"
              strokeWidth={selected?.regionId === region.id ? 3 : 2}
              onClick={(event) => { event.stopPropagation(); setSelected({ regionId: region.id, vertexIndex: null }); }}
            />
            {region.polygon.map(([x, y], vertexIndex) => (
              <circle
                key={`${region.id}-${vertexIndex}`}
                cx={x}
                cy={y}
                r={6}
                fill="var(--overlay-teal)"
                stroke="white"
                strokeWidth={2}
                tabIndex={disabled ? undefined : 0}
                role="button"
                aria-label={`영역 ${region.id} 꼭짓점 ${vertexIndex + 1}`}
                onFocus={() => setSelected({ regionId: region.id, vertexIndex })}
                onPointerDown={(event) => {
                  if (disabled) return;
                  event.stopPropagation();
                  event.currentTarget.setPointerCapture(event.pointerId);
                  setSelected({ regionId: region.id, vertexIndex });
                  dragRef.current = { regionId: region.id, vertexIndex, pointerId: event.pointerId };
                }}
              />
            ))}
          </g>
        ))}
        {draft ? (
          <g data-draft-region={draft.id}>
            <polyline points={pointsAttribute(draft.points)} fill="none" stroke="var(--overlay-teal)" strokeWidth={2} strokeDasharray="8 5" />
            {draft.points.map(([x, y], index) => <circle key={index} data-draft-point cx={x} cy={y} r={5} fill="var(--overlay-teal)" />)}
            <circle
              data-keyboard-cursor
              cx={keyboardCursor[0]}
              cy={keyboardCursor[1]}
              r={7}
              fill="none"
              stroke="white"
              strokeWidth={2}
              strokeDasharray="3 2"
            />
          </g>
        ) : null}
      </svg>
      <span id={instructionsId} className="sr-only">
        새 영역을 시작한 뒤 방향키로 위치를 옮기고 Enter 또는 Space로 점을 추가하세요.
      </span>

      <div className="absolute right-2 top-2 flex gap-1 rounded-control bg-card/90 p-1">
        <PrimaryToolButton
          label="직접 그리기"
          disabled={disabled || draft !== null || regions.length >= MAX_REGIONS}
          onClick={() => {
            setDraft({ id: generateUuidV4(), points: [] });
            setKeyboardCursor([imageWidth / 2, imageHeight / 2]);
            svgRef.current?.focus();
          }}
        />
        <PrimaryToolButton label="영역 완료" disabled={disabled || !draft || !validPolygon(draft.points)} onClick={closeDraft} />
        <ToolButton
          label="선택 영역 삭제"
          disabled={disabled || !selectedRegion}
          onClick={() => {
            if (!selectedRegion) return;
            onChange(regions.filter((region) => region.id !== selectedRegion.id).map((region) => ({ ...region, polygon: [...region.polygon] })));
            setSelected(null);
          }}
        ><Icon kind="delete" /></ToolButton>
        <ToolButton
          label="마지막 점 취소"
          disabled={disabled || !draft?.points.length}
          onClick={() => setDraft(draft ? { ...draft, points: draft.points.slice(0, -1) } : null)}
        ><Icon kind="undo" /></ToolButton>
      </div>

      {selectedPoint && selectedRegion && selectedVertexIndex !== null && selectedVertexIndex !== undefined ? (
        <div className="absolute bottom-2 right-2 flex gap-2 rounded-control bg-card/95 p-2 text-xs" role="group" aria-label="꼭짓점 좌표 편집">
          <label>X <input aria-label="꼭짓점 X" className="w-20 rounded border border-border bg-card px-1" type="number" min={0} max={imageWidth} value={selectedPoint[0]} disabled={disabled} onChange={(event) => replaceVertex(selectedRegion.id, selectedVertexIndex, [Math.min(imageWidth, Math.max(0, Number(event.target.value))), selectedPoint[1]])} /></label>
          <label>Y <input aria-label="꼭짓점 Y" className="w-20 rounded border border-border bg-card px-1" type="number" min={0} max={imageHeight} value={selectedPoint[1]} disabled={disabled} onChange={(event) => replaceVertex(selectedRegion.id, selectedVertexIndex, [selectedPoint[0], Math.min(imageHeight, Math.max(0, Number(event.target.value)))])} /></label>
        </div>
      ) : null}
    </div>
  );
}
