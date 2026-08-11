import { getPageLabel } from '@/shared/ui/NavBar';
import {
  useCamerasResource,
  useClipStorageResource,
  useConnectionResource,
  useDetectionSettingsResource,
  useStatusResource,
  useSystemResource,
} from '@/shared/api/usePollingResource';
import { CameraSection } from '@/features/settings/CameraSection';
import { DetectionSettingsCard } from '@/features/settings/DetectionSettingsCard';
import { EdgeSetupWizard } from '@/features/connection/EdgeSetupWizard';
import { ProcessingStatusCard } from '@/features/settings/ProcessingStatusCard';
import { ClipStorageCard } from '@/features/settings/ClipStorageCard';

/** 설정 화면 — 카메라(좌) + 탐지 설정/서버 연결/처리 상태/클립 저장 공간(우) 7:5 2컬럼 (front/design-handoff/README.md §5). */
export function SettingsPage(): JSX.Element {
  const camerasResource = useCamerasResource(true);
  const systemResource = useSystemResource(true);
  const detectionResource = useDetectionSettingsResource(true);
  const connectionResource = useConnectionResource(true);
  const statusResource = useStatusResource(true);
  const clipStorageResource = useClipStorageResource(true);

  return (
    <section className="page-placeholder">
      <h1 className="shell-page-title" tabIndex={-1} data-dialog-focus-fallback>
        {getPageLabel('settings')}
      </h1>

      <div className="mt-4 grid grid-cols-1 gap-5 lg:grid-cols-12">
        <div className="lg:col-span-7">
          <CameraSection camerasResource={camerasResource} systemResource={systemResource} />
        </div>

        <div className="flex flex-col gap-5 lg:col-span-5">
          <DetectionSettingsCard resource={detectionResource} />
          <EdgeSetupWizard connectionResource={connectionResource} camerasResource={camerasResource} />
          <ProcessingStatusCard resource={statusResource} />
          <ClipStorageCard resource={clipStorageResource} />
        </div>
      </div>
    </section>
  );
}
