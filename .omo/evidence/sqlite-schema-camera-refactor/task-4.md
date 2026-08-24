# Task 4 - Camera registration closes after create

## TDD

- Characterization GREEN before production edits: `CameraRegisterModal`, `CameraEditModal`, and `BedZoneRecognitionPanel` coverage passed (48 relevant assertions; the initial broad Vitest invocation also reported the repository suite green).
- RED: `CameraRegisterModal.test.tsx` required a successful create to emit the existing success toast, call the refresh callback once, call close once, and never render the retired step-2 live/bed-zone view. The unmodified implementation failed because it only advanced to step 2.
- GREEN: `pnpm --dir front exec vitest run src/features/settings/CameraRegisterModal.test.tsx src/features/settings/CameraEditModal.test.tsx src/features/settings/BedZoneRecognitionPanel.test.tsx` -> 3 files, 48 tests passed.

## Build

- `pnpm --dir front build` -> `tsc --noEmit && vite build` passed.
- `git diff --check` passed.

## Real Chromium fixture

A disposable stateful `/api/v1` fixture was driven through production `vite preview` at 1280x900. Browser waits subscribed to the POST response, modal visibility, row rendering, and polygon rendering; no timing sleeps were used.

1. Opened Settings and camera registration; used facility fixture `FAC-001`, space `101`, name `새 101호 카메라`, and RTSP `rtsp://10.0.0.20:554/stream`.
2. Submitted once. The modal disappeared, the refreshed registry showed the new row, and no registration bed-zone canvas/step rendered.
3. Edited that row. `BedZoneRecognitionPanel` rendered in `CameraEditModal`, and the recognized polygon was persisted/rendered.
4. Submitted `rtsp://bad.invalid/stream`; the API returned 422 and the registration modal remained visible with the established inline error.
5. The fixture recorded two creates total (one success, one intentional rejection), no duplicate success submission, and no unexpected console/page errors.

Artifacts:

- `task-4-registration-success.png`
- `task-4-edit-bed-zone-save.png`
- `task-4-registration-error.png`
- `task-4-browser-actions.json`

## Cleanup receipt

- Chromium closed by the QA script.
- Vite preview process stopped.
- Port 4173 confirmed free.
- `/tmp/seeon-task4-qa`, Vite log, and PID file removed.
