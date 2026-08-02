import { getEventTypeLabel, orderEventTypes } from '@/features/events/eventTypes';

type EventTypeFilterChipsProps = {
  totalCount: number;
  counts: ReadonlyMap<string, number>;
  selected: string | undefined;
  onSelect: (eventType: string | undefined) => void;
};

const CHIP_BASE = 'inline-flex h-9 items-center whitespace-nowrap rounded-full border px-4 text-sm font-semibold tabular-nums';
const CHIP_SELECTED = 'border-primary bg-primary text-primary-foreground';
const CHIP_UNSELECTED = 'border-border bg-card text-muted-foreground hover:bg-muted';

export function EventTypeFilterChips({ totalCount, counts, selected, onSelect }: EventTypeFilterChipsProps): JSX.Element {
  const types = orderEventTypes(counts.keys());

  return (
    <div className="flex flex-wrap items-center gap-2" role="group" aria-label="이벤트 종류 필터">
      <button
        type="button"
        className={`${CHIP_BASE} ${selected === undefined ? CHIP_SELECTED : CHIP_UNSELECTED}`}
        aria-pressed={selected === undefined}
        onClick={() => onSelect(undefined)}
      >
        전체 {totalCount}
      </button>
      {types.map((eventType) => (
        <button
          key={eventType}
          type="button"
          className={`${CHIP_BASE} ${selected === eventType ? CHIP_SELECTED : CHIP_UNSELECTED}`}
          aria-pressed={selected === eventType}
          onClick={() => onSelect(eventType)}
        >
          {getEventTypeLabel(eventType)} {counts.get(eventType) ?? 0}
        </button>
      ))}
    </div>
  );
}
