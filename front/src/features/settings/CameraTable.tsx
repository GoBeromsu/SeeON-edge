import { statusBadgeClassName, getCameraStatusMeta } from '@/shared/ui/StatusBadge';
import type { Camera } from '@/shared/api/client';

type CameraTableProps = {
  cameras: Camera[];
  onEdit: (camera: Camera) => void;
  onRequestDelete: (camera: Camera) => void;
};

function bedZoneBadge(camera: Camera): { label: string; className: string } {
  return camera.bed_zone
    ? { label: '인식 완료', className: statusBadgeClassName('approved') }
    : { label: '인식 필요', className: statusBadgeClassName('rejected') };
}

/** 설정 페이지 좌측 "카메라" 카드의 목록 테이블(front/design-handoff/README.md §5) — 카메라/층/침대 영역/상태/작업. */
export function CameraTable({ cameras, onEdit, onRequestDelete }: CameraTableProps): JSX.Element {
  if (cameras.length === 0) {
    return <p className="py-6 text-center text-sm text-muted-foreground">등록된 카메라가 없습니다.</p>;
  }

  return (
    <table className="w-full border-collapse text-sm">
      <thead>
        <tr className="border-b border-border text-left text-xs text-muted-foreground">
          <th className="py-2 pr-2 font-medium">카메라</th>
          <th className="py-2 pr-2 font-medium">층</th>
          <th className="py-2 pr-2 font-medium">침대 영역</th>
          <th className="py-2 pr-2 font-medium">상태</th>
          <th className="py-2 pl-2 text-right font-medium">작업</th>
        </tr>
      </thead>
      <tbody>
        {cameras.map((camera) => {
          const statusMeta = getCameraStatusMeta(camera.status);
          const bedZone = bedZoneBadge(camera);
          return (
            <tr
              key={camera.id}
              className={`border-b border-border last:border-0 ${camera.status === 'offline' ? 'bg-status-rejectedBg' : ''}`}
            >
              <td className="py-3 pr-2">
                <div className="font-medium text-foreground">{camera.label}</div>
                <div className="font-mono text-xs text-muted-foreground">{camera.rtsp_url_masked}</div>
              </td>
              <td className="py-3 pr-2 text-foreground">{camera.floor_name ?? '미지정'}</td>
              <td className="py-3 pr-2">
                <span className={bedZone.className}>{bedZone.label}</span>
              </td>
              <td className="py-3 pr-2">
                <span className={statusMeta.className}>
                  <span aria-hidden="true" className="h-1.5 w-1.5 rounded-full bg-current" />
                  {statusMeta.label}
                </span>
              </td>
              <td className="py-3 pl-2 text-right">
                <button
                  type="button"
                  className="text-sm font-semibold text-primary hover:underline"
                  onClick={() => onEdit(camera)}
                >
                  수정
                </button>
                <button
                  type="button"
                  className="ml-3 text-sm font-semibold text-destructive hover:underline"
                  onClick={() => onRequestDelete(camera)}
                >
                  삭제
                </button>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
