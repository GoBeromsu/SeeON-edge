import { act } from 'react';
import { createRoot } from 'react-dom/client';
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest';
import { AutoplayVideo } from '@/shared/ui/AutoplayVideo';

const activeRoots = new Set<ReturnType<typeof createRoot>>();

function renderVideo(src = '/api/v1/clips/clip-1/video') {
  const host = document.createElement('div');
  document.body.append(host);
  const root = createRoot(host);
  activeRoots.add(root);

  const rerender = (nextSrc: string): void => {
    act(() => root.render(
      <AutoplayVideo src={nextSrc} className="h-full w-full" onLoadedMetadata={vi.fn()} />,
    ));
  };

  rerender(src);
  return { host, rerender };
}

async function flushPlayback(): Promise<void> {
  await act(async () => Promise.resolve());
}

beforeEach(() => {
  vi.spyOn(HTMLMediaElement.prototype, 'play').mockResolvedValue();
  vi.spyOn(HTMLMediaElement.prototype, 'load').mockImplementation(() => undefined);
});

afterEach(() => {
  act(() => activeRoots.forEach((root) => root.unmount()));
  activeRoots.clear();
  document.body.innerHTML = '';
  vi.restoreAllMocks();
});

describe('AutoplayVideo', () => {
  it('shows autoplay-policy guidance only when play rejects with NotAllowedError', async () => {
    vi.mocked(HTMLMediaElement.prototype.play).mockRejectedValueOnce(new DOMException('blocked', 'NotAllowedError'));

    const { host } = renderVideo();
    await flushPlayback();

    expect(host.querySelector('[role="status"]')?.textContent).toContain('재생 버튼');
    expect(host.querySelector('[role="alert"]')).toBeNull();
  });

  it('shows a recoverable media failure when play rejects with NotSupportedError', async () => {
    vi.mocked(HTMLMediaElement.prototype.play).mockRejectedValueOnce(new DOMException('unsupported', 'NotSupportedError'));

    const { host } = renderVideo();
    await flushPlayback();

    expect(host.querySelector('[role="status"]')).toBeNull();
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('영상을 재생하지 못했습니다');
    expect(host.querySelector('button')?.textContent).toBe('다시 시도');
  });

  it('does not treat a non-DOM error named NotAllowedError as an autoplay-policy rejection', async () => {
    const rejection = new Error('playback failed');
    rejection.name = 'NotAllowedError';
    vi.mocked(HTMLMediaElement.prototype.play).mockRejectedValueOnce(rejection);

    const { host } = renderVideo();
    await flushPlayback();

    expect(host.querySelector('[role="status"]')).toBeNull();
    expect(host.querySelector('[role="alert"]')?.textContent).toContain('영상을 재생하지 못했습니다');
  });

  it('shows the media failure state when the video element reports a load or decode error', async () => {
    const { host } = renderVideo();
    await flushPlayback();

    act(() => host.querySelector('video')?.dispatchEvent(new Event('error')));

    expect(host.querySelector('[role="alert"]')?.textContent).toContain('영상을 재생하지 못했습니다');
    expect(host.querySelector('video')?.controls).toBe(true);
  });

  it('reloads and retries playback without removing native controls', async () => {
    const { host } = renderVideo();
    await flushPlayback();
    act(() => host.querySelector('video')?.dispatchEvent(new Event('error')));

    await act(async () => {
      host.querySelector<HTMLButtonElement>('button')?.click();
      await Promise.resolve();
    });

    expect(HTMLMediaElement.prototype.load).toHaveBeenCalledOnce();
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalledTimes(2);
    expect(host.querySelector('[role="alert"]')).toBeNull();
    expect(host.querySelector('video')?.controls).toBe(true);
  });

  it('clears a media failure after the video loads successfully', async () => {
    const { host } = renderVideo();
    await flushPlayback();
    const video = host.querySelector('video');
    act(() => video?.dispatchEvent(new Event('error')));

    act(() => video?.dispatchEvent(new Event('loadeddata')));

    expect(host.querySelector('[role="alert"]')).toBeNull();
  });

  it('clears stale failure state and retries autoplay when src changes', async () => {
    vi.mocked(HTMLMediaElement.prototype.play)
      .mockRejectedValueOnce(new DOMException('unsupported', 'NotSupportedError'))
      .mockResolvedValueOnce();
    const { host, rerender } = renderVideo();
    await flushPlayback();
    expect(host.querySelector('[role="alert"]')).not.toBeNull();

    rerender('/api/v1/clips/clip-2/video');
    await flushPlayback();

    expect(host.querySelector('[role="alert"]')).toBeNull();
    expect(host.querySelector('video')?.getAttribute('src')).toBe('/api/v1/clips/clip-2/video');
    expect(HTMLMediaElement.prototype.play).toHaveBeenCalledTimes(2);
  });
});
