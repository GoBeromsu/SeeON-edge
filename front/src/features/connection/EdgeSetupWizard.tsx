import { useEffect, useState } from 'react';
import type { CameraRegistry, ConnectionView } from '@/shared/api/client';
import { fetchTopology, fetchTopologyPreview, type CameraTopology, type TopologyPreview } from '@/shared/api/topologyClient';
import type { PollingResource } from '@/shared/api/usePollingResource';
import { ConnectionSettingsPanel } from '@/features/connection/ConnectionSettingsPanel';
import { CameraSyncStep } from '@/features/connection/CameraSyncStep';
import { ServerSyncStep } from '@/features/connection/ServerSyncStep';
import { WizardStep } from '@/features/connection/WizardStep';
import { deriveActiveStep, isCameraStepComplete, isDeviceStepComplete, isServerStepComplete } from '@/features/connection/wizardSteps';

type Props = {
  readonly connectionResource: PollingResource<ConnectionView>;
  readonly camerasResource: PollingResource<CameraRegistry>;
};

function TopologyUnavailable({ message, onRetry }: { readonly message: string | null; readonly onRetry: () => void }): JSX.Element {
  return (
    <div>
      <p role="alert" className="auth-error">{message ?? '카메라/토폴로지 상태를 불러오지 못했습니다.'}</p>
      <button type="button" className="dialog-secondary-action mt-3" onClick={onRetry}>다시 시도</button>
    </div>
  );
}

const STEP_TITLE: Record<1 | 2, string> = { 1: '장치 연결', 2: '카메라 확인' };

/**
 * A locked step must name the *earliest* unmet prerequisite, not merely the step immediately
 * before it -- otherwise an operator who is still on step 1 gets bounced from step 3 to step 2
 * ("2단계를 먼저 완료하세요"), lands on step 2 and finds it *also* locked, and has no single next
 * action. `blockingStep` is always `deriveActiveStep(...)` -- by construction the earliest step
 * that has not yet completed -- so any step after it can point straight at that one true blocker.
 */
function lockReason(blockingStep: 1 | 2): string {
  return `${blockingStep}단계 ${STEP_TITLE[blockingStep]}을 먼저 완료하세요.`;
}

/**
 * 3단계 Edge 설정 마법사.
 *   1단계 장치 연결 -- 기존 PUT /connection → test 흐름(ConnectionSettingsPanel)을 그대로 감싼다.
 *   2단계 카메라 확인 -- 이 장치가 아는 카메라 총계 + 층/방 배치 + 기존 sync-cameras 호출.
 *   3단계 서버 반영 -- 기존 topology preview/confirm 흐름을 그대로 감싼다.
 * 세 단계의 완료/잠금 여부는 오직 서버에서 읽은 상태로만 계산한다(wizardSteps.ts) -- 새로고침해도
 * React state나 localStorage가 아니라 서버가 실제로 아는 상태로 정확히 복귀해야 하기 때문이다.
 */
export function EdgeSetupWizard({ connectionResource, camerasResource }: Props): JSX.Element {
  const [topology, setTopology] = useState<CameraTopology | null>(null);
  const [preview, setPreview] = useState<TopologyPreview | null>(null);
  const [topologyLoading, setTopologyLoading] = useState(true);
  const [topologyError, setTopologyError] = useState<string | null>(null);

  async function reloadTopology(): Promise<void> {
    const [nextTopology, nextPreview] = await Promise.all([fetchTopology(), fetchTopologyPreview()]);
    setTopology(nextTopology);
    setPreview(nextPreview.preview);
    setTopologyError(null);
  }

  useEffect(() => {
    let active = true;
    setTopologyLoading(true);
    reloadTopology()
      .catch(() => { if (active) setTopologyError('카메라/토폴로지 상태를 불러오지 못했습니다.'); })
      .finally(() => { if (active) setTopologyLoading(false); });
    return () => { active = false; };
    // 최초 1회만 로드하며, 이후에는 각 단계의 액션(동기화/확정)이 명시적으로 reloadTopology를 호출한다.
  }, []);

  function retryTopology(): void {
    setTopologyLoading(true);
    reloadTopology()
      .catch(() => setTopologyError('카메라/토폴로지 상태를 불러오지 못했습니다.'))
      .finally(() => setTopologyLoading(false));
  }

  const connection = connectionResource.data;
  const cameras = camerasResource.data;

  const deviceComplete = isDeviceStepComplete(connection);
  const cameraComplete = isCameraStepComplete(cameras, topology);
  const serverComplete = isServerStepComplete(cameras, topology, preview);
  const activeStep = deriveActiveStep({ connection, cameras, topology, preview });

  return (
    <div className="flex flex-col gap-5">
      <WizardStep
        index={1}
        title="장치 연결"
        description="시설 코드와 일회용 등록 토큰을 입력하고 연결을 확인합니다."
        complete={deviceComplete}
        locked={false}
        lockReason=""
        active={activeStep === 1}
      >
        <ConnectionSettingsPanel resource={connectionResource} />
      </WizardStep>

      <WizardStep
        index={2}
        title="카메라 확인"
        description="이 장치가 알고 있는 카메라를 모두 확인한 뒤 서버와 동기화합니다."
        complete={cameraComplete}
        locked={activeStep < 2}
        lockReason={activeStep < 2 ? lockReason(activeStep as 1) : ''}
        active={activeStep === 2}
      >
        {topologyLoading ? (
          <p className="text-sm text-muted-foreground" aria-busy="true">카메라 상태를 확인하고 있습니다.</p>
        ) : topology ? (
          <CameraSyncStep camerasResource={camerasResource} topology={topology} onChanged={reloadTopology} />
        ) : (
          <TopologyUnavailable message={topologyError} onRetry={retryTopology} />
        )}
      </WizardStep>

      <WizardStep
        index={3}
        title="서버 반영"
        description="변경 사항을 확인하고 서버(Hub)에 반영합니다."
        complete={serverComplete}
        locked={activeStep < 3}
        lockReason={activeStep < 3 ? lockReason(activeStep as 1 | 2) : ''}
        active={activeStep === 3}
      >
        {topologyLoading ? (
          <p className="text-sm text-muted-foreground" aria-busy="true">서버 반영 상태를 확인하고 있습니다.</p>
        ) : topology ? (
          <ServerSyncStep cameras={cameras} topology={topology} preview={preview} onPreviewChanged={setPreview} onChanged={reloadTopology} />
        ) : (
          <TopologyUnavailable message={topologyError} onRetry={retryTopology} />
        )}
      </WizardStep>
    </div>
  );
}
