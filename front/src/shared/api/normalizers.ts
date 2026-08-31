export {
  normalizeBedZoneRecognitionResponse,
  normalizeCamera,
  normalizeCameraRegistry,
  normalizeCameraResponse,
  normalizeCameraTestResult,
} from '@/shared/api/cameraNormalizer';
export { normalizeClip, normalizeClipPageResponse, normalizeClipsResponse } from '@/shared/api/clipNormalizer';
export { normalizeClipScene } from '@/shared/api/clipSceneNormalizer';
export { normalizeClipStorageBrowse, normalizeClipStorageInfo } from '@/shared/api/clipStorageNormalizer';
export {
  normalizeConnectionTestResult,
  normalizeConnectionView,
} from '@/shared/api/connectionNormalizer';
export { normalizeDetectionSettings } from '@/shared/api/detectionSettingsNormalizer';
export { normalizeRuntimeSettings } from '@/shared/api/runtimeSettingsNormalizer';
export { isRecord } from '@/shared/api/normalizerFields';
export { normalizeStatusSnapshot } from '@/shared/api/statusNormalizer';
export { normalizeSystemSnapshot } from '@/shared/api/systemNormalizer';
