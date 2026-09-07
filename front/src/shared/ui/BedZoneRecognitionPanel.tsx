import { useEffect, useRef, useState } from 'react';
import { getCameraSnapshotUrl, recognizeBedZone, saveBedZone, type BedRegion, type BedZone } from '@/shared/api/client';
import { BedZonePolygonEditor } from '@/shared/ui/BedZonePolygonEditor';

type BedZoneRecognitionPanelProps = {
  cameraId: string;
  bedZone: BedZone | null;
  onSaved: (bedZone: BedZone | null) => void;
  onCancel: () => void;
};

function ActionButton({ label, disabled, onClick, primary = false }: { label: string; disabled?: boolean; onClick: () => void; primary?: boolean }): JSX.Element {
  return <button type="button" className={primary ? 'brand-action rounded-control px-4 py-2 text-sm font-semibold' : 'dialog-secondary-action'} aria-label={label} title={label} disabled={disabled} onClick={onClick}>{label}</button>;
}

export function BedZoneRecognitionPanel({ cameraId, bedZone, onSaved, onCancel }: BedZoneRecognitionPanelProps): JSX.Element {
  const [regions, setRegions] = useState<BedRegion[]>([]);
  const [dimensions, setDimensions] = useState<{ width: number; height: number } | null>(null);
  const [confidence, setConfidence] = useState(0.15);
  const [pending, setPending] = useState<'recognize' | 'save' | null>(null);
  const [editorValid, setEditorValid] = useState(true);
  const [status, setStatus] = useState<string | null>(null);
  const generationRef = useRef(0);
  const initialBedZoneRef = useRef(bedZone);
  initialBedZoneRef.current = bedZone;

  useEffect(() => {
    const initialBedZone = initialBedZoneRef.current;
    generationRef.current += 1;
    setRegions(initialBedZone?.regions.map((region) => ({ ...region, polygon: [...region.polygon] })) ?? []);
    setDimensions(initialBedZone ? { width: initialBedZone.image_width, height: initialBedZone.image_height } : null);
    setConfidence(0.15);
    setPending(null);
    setEditorValid(true);
    setStatus(null);
    return () => {
      generationRef.current += 1;
    };
  }, [cameraId]);

  async function recognize(): Promise<void> {
    if (pending) return;
    const generation = generationRef.current;
    setPending('recognize');
    setStatus(null);
    try {
      const candidate = await recognizeBedZone(cameraId, confidence);
      if (generation !== generationRef.current) return;
      setRegions(candidate.regions.map((region) => ({ ...region, polygon: [...region.polygon] })));
      setDimensions({ width: candidate.image_width, height: candidate.image_height });
      setStatus('침대 영역 후보 준비됨');
    } catch {
      if (generation === generationRef.current) setStatus('침대 영역 인식 실패');
    } finally {
      if (generation === generationRef.current) setPending(null);
    }
  }

  async function save(): Promise<void> {
    if (pending || !dimensions) return;
    const generation = generationRef.current;
    setPending('save');
    setStatus(null);
    try {
      const saved = await saveBedZone(cameraId, {
        regions,
        image_width: dimensions.width,
        image_height: dimensions.height,
      });
      if (generation !== generationRef.current) return;
      setStatus('저장됨 · 실시간 화면과 탐지에 반영되기까지 최대 1분');
      onSaved(saved);
    } catch {
      if (generation === generationRef.current) setStatus('침대 영역 저장 실패');
    } finally {
      if (generation === generationRef.current) setPending(null);
    }
  }

  const sensitivity = 1 - confidence;
  return <div>
    <div className="relative">
      <img
        src={getCameraSnapshotUrl(cameraId, 'bed-zone-editor')}
        alt="카메라 영상"
        className="block h-auto w-full"
        onLoad={(event) => {
          if (dimensions) return;
          const { naturalWidth, naturalHeight } = event.currentTarget;
          if (naturalWidth > 0 && naturalHeight > 0) setDimensions({ width: naturalWidth, height: naturalHeight });
        }}
      />
      {dimensions ? <BedZonePolygonEditor regions={regions} imageWidth={dimensions.width} imageHeight={dimensions.height} onChange={setRegions} onDraftValidityChange={setEditorValid} disabled={pending !== null} /> : null}
    </div>
    <p className="mt-2 text-sm text-muted-foreground">
      {editorValid ? '자동으로 찾거나 침대 모서리를 직접 지정하세요.' : '모서리를 찍은 뒤 영역 완료를 누르세요.'}
    </p>
    <p className="mt-1 text-sm font-medium text-foreground">침대 영역 {regions.length}개</p>
    {status ? (
      <div role={status.endsWith('실패') ? 'alert' : 'status'} aria-label={status} className={`mt-2 flex items-center gap-2 rounded-control border px-3 py-2 text-sm ${status.endsWith('실패') ? 'border-destructive/30 text-destructive' : 'border-border text-foreground'}`}>
        <span>{status}</span>
        {status.endsWith('실패') ? (
          <>
            <button type="button" className="font-semibold underline" aria-label="다시 시도" title="다시 시도" onClick={() => void (status === '침대 영역 저장 실패' ? save() : recognize())}>다시 시도</button>
            <span className="text-muted-foreground">직접 그리기도 사용할 수 있습니다.</span>
          </>
        ) : null}
      </div>
    ) : null}
    <div className="mt-3 space-y-3">
      <label className="flex items-center gap-2 text-sm font-medium text-foreground">
        <span>인식 민감도</span>
        <input aria-label="인식 민감도" title="인식 민감도" type="range" min={0.05} max={0.95} step={0.05} value={sensitivity} disabled={pending !== null} onChange={(event) => setConfidence(1 - Number(event.target.value))} />
      </label>
      <div className="flex flex-wrap justify-end gap-2">
        <ActionButton label="자동 인식" disabled={pending !== null} onClick={() => void recognize()} />
        <ActionButton label="저장" primary disabled={pending !== null || dimensions === null || !editorValid} onClick={() => void save()} />
        <ActionButton label="취소" disabled={pending !== null} onClick={onCancel} />
      </div>
    </div>
  </div>;
}
