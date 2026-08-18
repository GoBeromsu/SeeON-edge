import type { DetectionReason, RuntimeDetectionDiagnostics } from '@/shared/api/client';
import type { StatusVariant } from '@/shared/ui/StatusBadge';

export type DetectionState = RuntimeDetectionDiagnostics['state'];

/**
 * 감지 상태는 연결/매핑과 완전히 별개의 사실이다. 연결된 카메라가 감지를 못 하는
 * 상태(blind)가 실제로 존재하므로 두 표시는 서로를 가리거나 대체하지 않는다.
 * blind만 지속형 경보(`role="alert"`)이고 나머지는 일반 상태 표시로 남는다 —
 * 토스트가 아니라 화면에 붙어 있어야 근무 교대 뒤에도 사라지지 않는다.
 *
 * 라벨/톤은 호실 상세 카드와 관제 월 타일이 같은 표를 쓴다. 두 벌로 갈라지면
 * 같은 사실이 화면마다 다른 말로 보인다.
 */
const detectionMeta: Record<DetectionState, { label: string; variant: StatusVariant }> = {
  healthy: { label: '감지 정상', variant: 'approved' },
  starting: { label: '감지 상태 확인 중', variant: 'closed' },
  unknown: { label: '감지 상태 확인 불가', variant: 'pending' },
  disabled: { label: '감지 비활성', variant: 'closed' },
  blind: { label: '감지 중단', variant: 'rejected' },
};

/**
 * 줄바꿈이 의미 단위를 쪼개면 관제 중에 한 번에 읽히지 않는다. 붙어 있어야 하는
 * 명사구는 줄바꿈 없는 공백(\u00a0)으로 묶고, 문장 경계에서만 줄을 넘긴다.
 */
const blindGuidance: Record<DetectionReason, string> = {
  pose_not_completing: '자세\u00a0분석이 끝나지 않습니다. 카메라\u00a0화면과 처리\u00a0상태를 확인하세요.',
  decision_not_completing: '판단\u00a0단계가 끝나지 않습니다. 워커\u00a0처리\u00a0상태를 확인하세요.',
  no_completed_cycles: '완료된\u00a0감지가 한\u00a0건도 없습니다. 카메라\u00a0연결과 탐지\u00a0설정을 확인하세요.',
  telemetry_stale: '워커\u00a0보고가 지연되고 있습니다. 워커\u00a0상태를 확인하세요.',
  telemetry_missing: '워커\u00a0보고가 오지 않습니다. 워커\u00a0상태를 확인하세요.',
  counter_reset: '워커가 방금 재시작했습니다. 잠시\u00a0뒤 다시 확인하세요.',
};

const UNRESOLVED_BLIND_GUIDANCE = '감지가 진행되지 않습니다. 워커와 카메라\u00a0상태를 확인하세요.';

/** 진단이 없으면(구형 워커/미보고) 지어내지 않고 '확인 불가'로 남긴다. */
export function detectionStateOf(detection: RuntimeDetectionDiagnostics | undefined): DetectionState {
  return detection?.state ?? 'unknown';
}

export function detectionLabel(state: DetectionState): string {
  return detectionMeta[state].label;
}

export function detectionVariant(state: DetectionState): StatusVariant {
  return detectionMeta[state].variant;
}

export function blindGuidanceFor(reason: DetectionReason | null | undefined): string {
  return reason ? blindGuidance[reason] : UNRESOLVED_BLIND_GUIDANCE;
}

/**
 * 감지 상태 뱃지의 아이콘. 색만으로 뜻을 전하지 않기 위한 형태 신호이므로
 * blind일 때만 경고 삼각형, 나머지는 기존 상태 점을 그대로 쓴다. 장식용 움직임은 없다.
 */
export function DetectionStateIcon({ blind }: { blind: boolean }): JSX.Element {
  if (!blind) {
    return <span aria-hidden="true" className="h-1.5 w-1.5 shrink-0 rounded-full bg-current" />;
  }
  return (
    <svg
      aria-hidden="true"
      viewBox="0 0 12 12"
      className="h-3 w-3 shrink-0"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.4"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M6 1.4 11 10.6H1L6 1.4Z" />
      <path d="M6 5v2.2" />
      <path d="M6 9.1h.01" />
    </svg>
  );
}
