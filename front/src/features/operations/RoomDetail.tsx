import { useState } from 'react';
import { LiveStreamPanel } from '@/features/operations/LiveStreamPanel';
import { CameraInfoCard } from '@/features/operations/CameraInfoCard';
import { DetectionSettingsCard } from '@/features/operations/DetectionSettingsCard';
import { EventHistoryList } from '@/features/operations/EventHistoryList';
import { ClipPlayerModal } from '@/features/operations/ClipPlayerModal';
import { useStatusResource } from '@/shared/api/usePollingResource';
import type { Camera, Clip } from '@/shared/api/client';

type RoomDetailProps = {
  camera: Camera;
  onBack: () => void;
};

export function RoomDetail({ camera, onBack }: RoomDetailProps): JSX.Element {
  const { data: status } = useStatusResource(true);
  const diagnostics = status?.runtime.cameras[camera.id];
  const [selectedClip, setSelectedClip] = useState<Clip | null>(null);

  return (
    <section aria-label="호실 상세">
      <div className="mb-4 flex items-center gap-3">
        <button
          type="button"
          onClick={onBack}
          className="min-h-11 rounded-control border border-border px-3 text-sm font-semibold text-foreground"
        >
          ← 카메라 월로
        </button>
        <h2 className="text-lg font-semibold text-foreground">{camera.label}</h2>
      </div>

      <div className="grid grid-cols-1 gap-6 lg:grid-cols-3">
        <div className="lg:col-span-2">
          <LiveStreamPanel camera={camera} diagnostics={diagnostics} />
        </div>
        <div className="flex flex-col gap-4 lg:col-span-1">
          <CameraInfoCard camera={camera} />
          <DetectionSettingsCard cameraId={camera.id} />
        </div>
      </div>

      <div className="mt-8">
        <EventHistoryList cameraId={camera.id} onSelectClip={setSelectedClip} />
      </div>

      <ClipPlayerModal clip={selectedClip} cameraLabel={camera.label} onClose={() => setSelectedClip(null)} />
    </section>
  );
}
