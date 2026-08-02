import type { Camera } from '@/shared/api/types';

type CameraFilterSelectProps = {
  cameras: readonly Camera[];
  value: string;
  onChange: (cameraId: string) => void;
  className?: string;
};

export function CameraFilterSelect({ cameras, value, onChange, className = '' }: CameraFilterSelectProps): JSX.Element {
  return (
    <label className={`inline-flex items-center gap-2 text-sm font-semibold text-muted-foreground ${className}`}>
      카메라
      <select
        value={value}
        onChange={(event) => onChange(event.target.value)}
        className="h-9 rounded-control border border-input bg-card px-3 text-sm font-medium text-foreground"
      >
        <option value="">전체</option>
        {cameras.map((camera) => (
          <option key={camera.id} value={camera.id}>{camera.label}</option>
        ))}
      </select>
    </label>
  );
}
