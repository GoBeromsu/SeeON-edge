import type { Camera, Clip } from '@/shared/api/types';

/**
 * The `/clips` API only returns `camera_id` (backend ClipManifestResponse has no camera_label
 * field), so clipNormalizer.normalizeClip's camera_label always falls back to its generic "카메라
 * 미상" placeholder. Resolve the real label from the cameras resource instead; that placeholder
 * (or whatever clipNormalizer produced) remains the last-resort fallback for a clip whose camera
 * has since been deleted.
 */
export function resolveCameraLabel(cameras: readonly Camera[], clip: Clip): string {
  const match = clip.camera_id ? cameras.find((camera) => camera.id === clip.camera_id) : undefined;
  return match?.label ?? clip.camera_label;
}
