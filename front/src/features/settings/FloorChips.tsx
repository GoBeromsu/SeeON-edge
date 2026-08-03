import { useState } from 'react';

type FloorChipsProps = {
  options: string[];
  selected: string | null;
  onSelect: (floor: string | null) => void;
  onAddFloor: (floor: string) => void;
};

const CHIP_BASE = 'inline-flex h-8 items-center whitespace-nowrap rounded-full border px-3 text-sm';
const CHIP_SELECTED = 'border-primary/30 bg-primary/10 font-semibold text-primary';
const CHIP_UNSELECTED = 'border-border bg-card text-muted-foreground hover:bg-muted';

/**
 * 층 선택 칩 + "+ 층 추가" (front/design-handoff/Eldercare Prototype.dc.html:291-297, :327-333, issue #85).
 * 이미 선택된 칩을 다시 누르면 선택이 해제된다(null) — 로컬 override를 지워 space-sync floor_name로
 * 되돌리는 동작과 대응한다(카메라 add/edit 모달의 handleSave 참고).
 */
export function FloorChips({ options, selected, onSelect, onAddFloor }: FloorChipsProps): JSX.Element {
  const [adding, setAdding] = useState(false);
  const [draft, setDraft] = useState('');

  function commitDraft(): void {
    const value = draft.trim();
    if (value) {
      onAddFloor(value);
    }
    setDraft('');
    setAdding(false);
  }

  return (
    <div className="flex flex-wrap items-center gap-1.5" role="group" aria-label="층 선택">
      {options.map((floor) => (
        <button
          key={floor}
          type="button"
          className={`${CHIP_BASE} ${selected === floor ? CHIP_SELECTED : CHIP_UNSELECTED}`}
          aria-pressed={selected === floor}
          onClick={() => onSelect(selected === floor ? null : floor)}
        >
          {floor}
        </button>
      ))}
      {adding ? (
        <input
          autoFocus
          className="h-8 w-24 rounded-full border border-input bg-background px-3 text-sm"
          value={draft}
          placeholder="예: 3층"
          aria-label="새 층 이름"
          onChange={(event) => setDraft(event.target.value)}
          onKeyDown={(event) => {
            if (event.key === 'Enter') {
              event.preventDefault();
              commitDraft();
            } else if (event.key === 'Escape') {
              setDraft('');
              setAdding(false);
            }
          }}
          onBlur={commitDraft}
        />
      ) : (
        <button
          type="button"
          className={`${CHIP_BASE} border-dashed text-muted-foreground`}
          onClick={() => setAdding(true)}
        >
          + 층 추가
        </button>
      )}
    </div>
  );
}
