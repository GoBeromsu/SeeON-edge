import type { Camera } from '@/shared/api/types';

type CameraFilterSelectProps = {
  cameras: readonly Camera[];
  value: string;
  onChange: (cameraId: string) => void;
  className?: string;
};

export function CameraFilterSelect({ cameras, value, onChange, className = '' }: CameraFilterSelectProps): JSX.Element {
  return (
    <select
      aria-label="카메라"
      value={value}
      onChange={(event) => onChange(event.target.value)}
      className={`h-9 min-h-11 rounded-control border border-input bg-card px-3 text-sm font-medium text-foreground ${className}`}
    >
      <option value="">전체 카메라</option>
      {cameras.map((camera) => (
        <option key={camera.id} value={camera.id}>{camera.label}</option>
      ))}
    </select>
  );
}
