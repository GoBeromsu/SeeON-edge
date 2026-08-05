import { FLOOR_VALUES, floorLabel } from '@/features/settings/floorOptions';

type FloorSelectProps = {
  value: number | null;
  onChange: (floor: number | null) => void;
  /**
   * Renders a leading "clear the override" option when set (even to `null`) -- used only by the
   * edit modal, where a camera can already have a space-sync `floor_name` to fall back to.
   * Registration never passes this: a brand-new camera has no `floor_name` yet, so there is
   * nothing meaningful to fall back to.
   */
  unsetLabel?: string | null;
  disabled?: boolean;
};

/**
 * 층 선택 드롭다운 (issue #155): 자유 문자열 칩 대신 고정 목록(B1~10층)에서 고른다. 정수로
 * 저장하고 표시 문자열은 렌더 시점에 만든다 (`floorLabel`) -- 이전에는 같은 층이 '2층'/'2'/'B1'/
 * '2 층'처럼 저장 시점에 자유롭게 입력돼 값이 갈렸다.
 */
export function FloorSelect({ value, onChange, unsetLabel, disabled }: FloorSelectProps): JSX.Element {
  return (
    <select
      name="floor"
      aria-label="층"
      value={value === null ? '' : String(value)}
      disabled={disabled}
      onChange={(event) => {
        const raw = event.target.value;
        onChange(raw === '' ? null : Number(raw));
      }}
    >
      {unsetLabel !== undefined ? (
        <option value="">{unsetLabel ? `클라우드 값 사용 (${unsetLabel})` : '미지정'}</option>
      ) : null}
      {FLOOR_VALUES.map((floor) => (
        <option key={floor} value={floor}>
          {floorLabel(floor)}
        </option>
      ))}
    </select>
  );
}
