import { useState } from 'react';
import { getClipThumbnailUrl } from '@/shared/api/session';
import type { Clip } from '@/shared/api/types';

type ClipThumbnailProps = {
  clip: Clip;
  alt: string;
};

export function ClipThumbnail({ clip, alt }: ClipThumbnailProps): JSX.Element {
  const [failedClip, setFailedClip] = useState<Clip | null>(null);
  const loadFailed = failedClip === clip;

  if (clip.thumbnail_available && !loadFailed) {
    return (
      <img
        src={getClipThumbnailUrl(clip.id)}
        alt={alt}
        width={640}
        height={360}
        loading="lazy"
        className="h-full w-full object-contain"
        onLoad={() => setFailedClip(null)}
        onError={() => setFailedClip(clip)}
      />
    );
  }

  if (clip.video_available) {
    return (
      <div className="flex h-full items-center justify-center" data-testid="clip-thumbnail-available" aria-hidden="true">
        <svg className="h-8 w-8" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75">
          <path d="M8.5 6.75v10.5L17 12 8.5 6.75Z" strokeLinejoin="round" />
        </svg>
      </div>
    );
  }

  return (
    <div className="event-media-unavailable h-full px-3 text-center text-xs" data-testid="clip-thumbnail-unavailable">
      {clip.video_error ?? '영상을 사용할 수 없습니다.'}
    </div>
  );
}
