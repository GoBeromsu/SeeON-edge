const EVENT_FACETS = ['fall', 'bed-exit', 'other'] as const;

export type EventFacet = (typeof EVENT_FACETS)[number];

export function toEventFacet(rawEventType: string | null | undefined): EventFacet {
  switch (rawEventType) {
    case 'fall':
      return 'fall';
    case 'bed-exit':
      return 'bed-exit';
    default:
      return 'other';
  }
}
