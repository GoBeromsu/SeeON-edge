import { useEffect, useRef, useState } from 'react';
import { browseClipStorage, type ClipStorageBrowseResult } from '@/shared/api/client';
import { AccessibleDialog } from '@/shared/ui/AccessibleDialog';

type FolderBrowserModalProps = {
  open: boolean;
  initialPath: string;
  onClose: () => void;
  onSelect: (path: string) => void;
};

function FolderIcon(): JSX.Element {
  return (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth={1.75} strokeLinecap="round" strokeLinejoin="round" className="h-4 w-4 shrink-0">
      <path d="M4 6a1 1 0 0 1 1-1h4l2 2h8a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1Z" />
    </svg>
  );
}

/** 클립 저장 공간 카드의 "폴더 탐색기" 모달 (front/design-handoff/README.md §5 모달 4). */
export function FolderBrowserModal({ open, initialPath, onClose, onSelect }: FolderBrowserModalProps): JSX.Element {
  const [path, setPath] = useState(initialPath);
  const [result, setResult] = useState<ClipStorageBrowseResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const wasOpen = useRef(open);
  const requestId = useRef(0);

  useEffect(() => {
    if (open && !wasOpen.current) {
      setPath(initialPath);
    }
    wasOpen.current = open;
  }, [open, initialPath]);

  useEffect(() => {
    if (!open) return;
    const id = ++requestId.current;
    setLoading(true);
    setError(null);
    browseClipStorage(path).then(
      (next) => {
        if (id !== requestId.current) return;
        setResult(next);
        setLoading(false);
      },
      () => {
        if (id !== requestId.current) return;
        setError('폴더 목록을 불러오지 못했습니다.');
        setLoading(false);
      },
    );
  }, [open, path]);

  return (
    <AccessibleDialog open={open} title="폴더 탐색기" onClose={onClose}>
      <div className="flex items-center gap-2 border-b border-border pb-2">
        <button
          type="button"
          className="icon-button"
          aria-label="상위 폴더로 이동"
          disabled={!result?.parent}
          onClick={() => result?.parent !== null && result?.parent !== undefined && setPath(result.parent)}
        >
          ↑
        </button>
        <span className="truncate font-mono text-xs text-foreground">{result?.path ?? path}</span>
      </div>

      <div className="mt-2 max-h-64 min-h-[6rem] overflow-auto">
        {loading ? (
          <p className="py-4 text-center text-sm text-muted-foreground">폴더를 불러오는 중입니다...</p>
        ) : error ? (
          <p role="alert" className="py-4 text-center text-sm text-destructive">{error}</p>
        ) : result && result.directories.length === 0 ? (
          <p className="py-4 text-center text-sm text-muted-foreground">하위 폴더 없음</p>
        ) : (
          <ul>
            {result?.directories.map((entry) => (
              <li key={entry.path}>
                <button
                  type="button"
                  className="flex w-full items-center gap-2 rounded-control px-2 py-2 text-left text-sm text-foreground hover:bg-muted"
                  onClick={() => setPath(entry.path)}
                >
                  <FolderIcon />
                  {entry.name}
                </button>
              </li>
            ))}
          </ul>
        )}
      </div>

      <div className="dialog-actions mt-3">
        <button type="button" className="dialog-secondary-action" onClick={onClose}>
          취소
        </button>
        <button
          type="button"
          className="brand-action inline-flex h-9 items-center justify-center rounded-control px-4 text-sm font-semibold"
          disabled={!result}
          onClick={() => result && onSelect(result.path)}
        >
          이 위치 사용
        </button>
      </div>
    </AccessibleDialog>
  );
}
