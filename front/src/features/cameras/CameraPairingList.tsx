import { useState } from 'react';
import { updateCamera, type CameraRegistry } from '@/shared/api/client';
import type { CameraTopology } from '@/shared/api/topologyClient';
import type { PollingResource } from '@/shared/api/usePollingResource';

type Props = {
  readonly topology: CameraTopology;
  readonly camerasResource: PollingResource<CameraRegistry>;
  readonly onChanged: () => Promise<void>;
  readonly onError: (message: string) => void;
};

export function CameraPairingList({ topology, camerasResource, onChanged, onError }: Props): JSX.Element {
  const [busyId, setBusyId] = useState<string | null>(null);
  const rooms = topology.floors.flatMap((floor) => floor.rooms.map((room) => ({ room, floor })));
  const cameras = (camerasResource.data?.cameras ?? []).filter((camera) => topology.unmapped_camera_ids.includes(camera.id));

  async function pair(cameraId: string, form: HTMLFormElement): Promise<void> {
    if (busyId) return;
    const data = new FormData(form);
    const edgeRef = String(data.get(`camera_ref_${cameraId}`) ?? '').trim();
    const roomRef = String(data.get(`camera_room_${cameraId}`) ?? '').trim();
    if (!edgeRef || !roomRef) {
      onError('카메라 참조와 방을 모두 선택하세요.');
      return;
    }
    setBusyId(cameraId);
    try {
      await updateCamera(cameraId, { edge_ref: edgeRef, room_edge_ref: roomRef });
      camerasResource.retry();
      await onChanged();
    } catch {
      onError('카메라 매핑을 저장하지 못했습니다. 중복 참조와 방 점유 상태를 확인하세요.');
    } finally {
      setBusyId(null);
    }
  }

  if (!cameras.length) return <p className="text-sm text-status-approvedFg">모든 카메라의 층/방 매핑이 완료되었습니다.</p>;
  return (
    <div className="space-y-2">
      {cameras.map((camera) => (
        <form key={camera.id} className="grid gap-2 rounded-control bg-muted p-3 sm:grid-cols-[minmax(0,1fr)_minmax(9rem,1fr)_auto]" onSubmit={(event) => { event.preventDefault(); void pair(camera.id, event.currentTarget); }}>
          <label className="grid gap-1 text-xs font-semibold text-muted-foreground">{camera.label} 카메라 참조<input className="h-11 min-w-0 rounded-control border border-input bg-background px-3 text-sm sm:h-9" name={`camera_ref_${camera.id}`} placeholder="camera-hall" disabled={busyId !== null} /></label>
          <label className="grid gap-1 text-xs font-semibold text-muted-foreground">설치 방<select className="h-11 min-w-0 rounded-control border border-input bg-background px-3 text-sm sm:h-9" name={`camera_room_${camera.id}`} disabled={busyId !== null}><option value="">방 선택</option>{rooms.map(({ floor, room }) => <option key={room.edge_ref} value={room.edge_ref}>{floor.name} · {room.name}</option>)}</select></label>
          <button className="brand-action mt-auto h-11 rounded-control px-3 text-sm font-semibold sm:h-9" disabled={busyId !== null}>{busyId === camera.id ? '저장 중...' : '매핑 저장'}</button>
        </form>
      ))}
    </div>
  );
}
