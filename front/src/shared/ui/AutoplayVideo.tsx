import { useEffect, useRef, useState } from 'react';

type AutoplayVideoProps = {
  src: string;
  className: string;
  onLoadedMetadata: (video: HTMLVideoElement) => void;
  onVideoElement?: (video: HTMLVideoElement | null) => void;
};

type PlaybackFailure = 'blocked' | 'failed' | null;

function classifyPlaybackFailure(error: unknown): Exclude<PlaybackFailure, null> {
  return error instanceof DOMException && error.name === 'NotAllowedError'
    ? 'blocked'
    : 'failed';
}

export function AutoplayVideo({ src, className, onLoadedMetadata, onVideoElement }: AutoplayVideoProps): JSX.Element {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const [failure, setFailure] = useState<PlaybackFailure>(null);

  useEffect(() => {
    const video = videoRef.current;
    if (!video) return undefined;

    let mounted = true;
    setFailure(null);
    void video.play().then(
      () => {
        if (mounted) {
          setFailure(null);
        }
      },
      (error: unknown) => {
        if (mounted) {
          const failure = classifyPlaybackFailure(error);
          setFailure(failure);
        }
      },
    );
    return () => {
      mounted = false;
    };
  }, [src]);

  const retryPlayback = (): void => {
    const video = videoRef.current;
    if (!video) return;

    setFailure(null);
    video.load();
    void video.play().then(
      () => {
        setFailure(null);
      },
      (error: unknown) => {
        const failure = classifyPlaybackFailure(error);
        setFailure(failure);
      },
    );
  };

  return (
    <>
      <video
        ref={(video) => {
          videoRef.current = video;
          onVideoElement?.(video);
        }}
        className={className}
        src={src}
        controls
        autoPlay
        playsInline
        preload="metadata"
        onError={() => {
          setFailure('failed');
        }}
        onLoadedData={() => setFailure(null)}
        onPlay={() => setFailure(null)}
        onLoadedMetadata={(event) => onLoadedMetadata(event.currentTarget)}
      />
      {failure === 'blocked' ? (
        <p className="media-status-overlay pointer-events-none absolute inset-x-2 top-2 rounded-control px-3 py-2 text-sm" role="status">
          자동 재생이 차단되었습니다. 재생 버튼을 눌러 영상을 확인하세요.
        </p>
      ) : null}
      {failure === 'failed' ? (
        <div className="media-status-overlay absolute inset-x-2 top-2 flex flex-col items-start gap-3 rounded-control px-3 py-2 text-sm sm:flex-row sm:items-center sm:justify-between" role="alert">
          <span>영상을 재생하지 못했습니다.</span>
          <button
            type="button"
            className="min-h-11 shrink-0 rounded-control border border-border bg-card px-3 text-xs font-semibold text-foreground hover:bg-muted"
            onClick={retryPlayback}
          >
            다시 시도
          </button>
        </div>
      ) : null}
    </>
  );
}
