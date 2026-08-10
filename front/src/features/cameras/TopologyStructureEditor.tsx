import { FormEvent, useState } from 'react';
import {
  createTopologyFloor,
  createTopologyRoom,
  updateTopologyFloor,
  updateTopologyRoom,
  type CameraTopology,
} from '@/shared/api/topologyClient';

type Props = {
  readonly topology: CameraTopology;
  readonly onChanged: () => Promise<void>;
  readonly onError: (message: string) => void;
};

const inputClass = 'h-11 min-w-0 rounded-control border border-input bg-background px-3 text-sm text-foreground sm:h-9';

function value(form: HTMLFormElement, name: string): string {
  return String(new FormData(form).get(name) ?? '').trim();
}

export function TopologyStructureEditor({ topology, onChanged, onError }: Props): JSX.Element {
  const [busy, setBusy] = useState(false);

  async function mutate(action: () => Promise<unknown>, success: () => void): Promise<void> {
    if (busy) return;
    setBusy(true);
    onError('');
    try {
      await action();
      success();
      await onChanged();
    } catch {
      onError('층/방 정보를 저장하지 못했습니다. 중복 참조와 상위 항목을 확인하세요.');
    } finally {
      setBusy(false);
    }
  }

  function addFloor(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    const form = event.currentTarget;
    const edgeRef = value(form, 'floor_edge_ref');
    const name = value(form, 'floor_name');
    const order = Number(value(form, 'floor_order'));
    if (!edgeRef || !name || !Number.isInteger(order) || order < 0) {
      onError('층 참조, 이름, 순서를 모두 입력하세요.');
      return;
    }
    void mutate(() => createTopologyFloor({ edge_ref: edgeRef, name, order_index: order }), () => form.reset());
  }

  function addRoom(event: FormEvent<HTMLFormElement>): void {
    event.preventDefault();
    const form = event.currentTarget;
    const edgeRef = value(form, 'room_edge_ref');
    const floorRef = value(form, 'room_floor_ref');
    const name = value(form, 'room_name');
    const legacyId = value(form, 'legacy_canonical_space_id');
    if (!edgeRef || !floorRef || !name) {
      onError('방 참조, 상위 층, 이름을 모두 입력하세요.');
      return;
    }
    void mutate(() => createTopologyRoom({ edge_ref: edgeRef, floor_edge_ref: floorRef, name,
      legacy_canonical_space_id: legacyId || null }), () => form.reset());
  }

  return (
    <div className="space-y-4">
      <form className="grid gap-2 sm:grid-cols-[1fr_1fr_5rem_auto]" onSubmit={addFloor}>
        <label className="grid gap-1 text-xs font-semibold text-muted-foreground">층 참조<input className={inputClass} name="floor_edge_ref" placeholder="floor-1" disabled={busy} /></label>
        <label className="grid gap-1 text-xs font-semibold text-muted-foreground">층 이름<input className={inputClass} name="floor_name" placeholder="1층" disabled={busy} /></label>
        <label className="grid gap-1 text-xs font-semibold text-muted-foreground">순서<input className={inputClass} name="floor_order" type="number" min="0" defaultValue="1" disabled={busy} /></label>
        <button className="brand-action mt-auto h-11 rounded-control px-3 text-sm font-semibold sm:h-9" disabled={busy}>층 추가</button>
      </form>

      {topology.floors.map((floor) => (
        <section key={floor.edge_ref} className="rounded-control border border-border p-3">
          <form className="grid gap-2 sm:grid-cols-[1fr_5rem_auto]" onSubmit={(event) => {
            event.preventDefault();
            const form = event.currentTarget;
            void mutate(() => updateTopologyFloor(floor.edge_ref, { name: value(form, 'name'), order_index: Number(value(form, 'order')) }), () => undefined);
          }}>
            <label className="grid gap-1 text-xs font-semibold text-muted-foreground">{floor.edge_ref}<input className={inputClass} name="name" defaultValue={floor.name} disabled={busy} /></label>
            <label className="grid gap-1 text-xs font-semibold text-muted-foreground">순서<input className={inputClass} name="order" type="number" min="0" defaultValue={floor.order_index} disabled={busy} /></label>
            <button className="dialog-secondary-action mt-auto !h-11 sm:!h-9" disabled={busy}>층 저장</button>
          </form>
          <div className="mt-3 space-y-2 border-t border-border pt-3">
            {floor.rooms.map((room) => (
              <form key={room.edge_ref} className="flex flex-wrap items-end gap-2" onSubmit={(event) => {
                event.preventDefault();
                const form = event.currentTarget;
                void mutate(() => updateTopologyRoom(room.edge_ref, value(form, 'name')), () => undefined);
              }}>
                <label className="grid min-w-[12rem] flex-1 gap-1 text-xs font-semibold text-muted-foreground">{room.edge_ref}<input className={inputClass} name="name" defaultValue={room.name} disabled={busy} /></label>
                <span className="pb-2 text-xs text-muted-foreground">카메라 {room.cameras.length}대</span>
                <button className="dialog-secondary-action !h-11 sm:!h-9" disabled={busy}>방 저장</button>
              </form>
            ))}
          </div>
        </section>
      ))}

      <form className="grid gap-2 border-t border-border pt-4 sm:grid-cols-2" onSubmit={addRoom}>
        <label className="grid gap-1 text-xs font-semibold text-muted-foreground">방 참조<input className={inputClass} name="room_edge_ref" placeholder="room-101" disabled={busy} /></label>
        <label className="grid gap-1 text-xs font-semibold text-muted-foreground">상위 층<select className={inputClass} name="room_floor_ref" disabled={busy}><option value="">층 선택</option>{topology.floors.map((floor) => <option key={floor.edge_ref} value={floor.edge_ref}>{floor.name}</option>)}</select></label>
        <label className="grid gap-1 text-xs font-semibold text-muted-foreground">방 이름<input className={inputClass} name="room_name" placeholder="101호" disabled={busy} /></label>
        <label className="grid gap-1 text-xs font-semibold text-muted-foreground">기존 공간 ID (선택)<input className={inputClass} name="legacy_canonical_space_id" disabled={busy} /></label>
        <button className="brand-action h-11 rounded-control px-3 text-sm font-semibold sm:col-start-2 sm:h-9 sm:justify-self-end" disabled={busy}>방 추가</button>
      </form>
    </div>
  );
}
