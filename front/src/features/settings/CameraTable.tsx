import { floorLabel } from '@/features/settings/floorOptions';
import { getCameraStatusMeta } from '@/shared/ui/StatusBadge';
import type { Camera } from '@/shared/api/client';

type CameraTableProps = {
  cameras: Camera[];
  onEdit: (camera: Camera) => void;
  onRequestDelete: (camera: Camera) => void;
};

/**
 * 설정 페이지 좌측 "카메라" 카드의 목록 테이블 (front/design-handoff/Eldercare Prototype.dc.html:179-191)
 * — 카메라/층/상태/작업 4열. 이전에는 "침대 영역" 열이 있었지만 핸드오프에는 없다(issue #85) — 기능은
 * 잃지 않고 편집 모달("수정" → CameraEditModal)의 침대 영역 뱃지/재인식 버튼으로 이동했다.
 */
export function CameraTable({ cameras, onEdit, onRequestDelete }: CameraTableProps): JSX.Element {
  if (cameras.length === 0) {
    return <p className="py-6 text-center text-sm text-muted-foreground">등록된 카메라가 없습니다.</p>;
  }

  return (
    // table-fixed + break-all keeps long masked RTSPS URLs from expanding body.scrollWidth on narrow viewports (QA F1).
    <table className="w-full table-fixed border-collapse text-sm">
      <thead>
        <tr className="border-b border-border text-left text-xs text-muted-foreground">
          <th className="min-w-0 py-2 pr-2 font-medium">카메라</th>
          <th className="w-12 py-2 pr-2 font-medium whitespace-nowrap sm:w-16">층</th>
          <th className="w-[5.75rem] py-2 pr-2 font-medium whitespace-nowrap sm:w-28">상태</th>
          <th className="w-12 py-2 pl-2 text-right font-medium whitespace-nowrap sm:w-[4.5rem]">작업</th>
        </tr>
      </thead>
      <tbody>
        {cameras.map((camera) => {
          const statusMeta = getCameraStatusMeta(camera.status);
          return (
            <tr
              key={camera.id}
              className={`border-b border-border last:border-0 ${camera.status === 'offline' ? 'bg-status-rejectedBg' : ''}`}
            >
              <td className="min-w-0 py-3 pr-2 align-top">
                <div className="break-words font-medium text-foreground">{camera.label}</div>
                <div className="break-all font-mono text-xs text-muted-foreground">{camera.rtsp_url_masked}</div>
              </td>
              <td className="py-3 pr-2 align-top whitespace-nowrap text-foreground">
                {camera.floor !== null && camera.floor !== undefined
                  ? floorLabel(camera.floor)
                  : camera.floor_name ?? '미지정'}
              </td>
              <td className="py-3 pr-2 align-top whitespace-nowrap">
                <span className={statusMeta.className}>
                  <span aria-hidden="true" className="h-1.5 w-1.5 shrink-0 rounded-full bg-current" />
                  {statusMeta.label}
                </span>
              </td>
              <td className="py-3 pl-2 align-top text-right whitespace-nowrap">
                <div className="inline-flex flex-col items-end gap-1 sm:flex-row sm:items-center sm:gap-0">
                  <button
                    type="button"
                    className="text-sm font-semibold text-primary hover:underline"
                    onClick={() => onEdit(camera)}
                  >
                    수정
                  </button>
                  <button
                    type="button"
                    className="text-sm font-semibold text-destructive hover:underline sm:ml-3"
                    onClick={() => onRequestDelete(camera)}
                  >
                    삭제
                  </button>
                </div>
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
