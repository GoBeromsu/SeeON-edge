export const EVENTS_PAGE_SIZE = 48;
export const EVENTS_POLL_INTERVAL_MS = 8_000;

export type PagerItem =
  | { readonly kind: 'page'; readonly pageIndex: number }
  | { readonly kind: 'ellipsis'; readonly key: string };

export function getPageCount(total: number): number {
  return Math.max(1, Math.ceil(total / EVENTS_PAGE_SIZE));
}

export function getPagerItems(pageIndex: number, pageCount: number): readonly PagerItem[] {
  const visiblePages = new Set([0, pageCount - 1, pageIndex - 1, pageIndex, pageIndex + 1]);
  const pages = [...visiblePages]
    .filter((candidate) => candidate >= 0 && candidate < pageCount)
    .sort((left, right) => left - right);
  const items: PagerItem[] = [];
  for (const page of pages) {
    const previous = items.at(-1);
    if (previous?.kind === 'page' && page - previous.pageIndex > 1) {
      items.push({ kind: 'ellipsis', key: `${previous.pageIndex}-${page}` });
    }
    items.push({ kind: 'page', pageIndex: page });
  }
  return items;
}
