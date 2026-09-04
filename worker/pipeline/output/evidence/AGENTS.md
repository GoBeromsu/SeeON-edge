# worker/pipeline/output/evidence

Own the evidence path after alert admission: smart record actor, clip
publication, sealed sidecar, durable stager, delivery queue, and snapshot store.
The Flow media plane supplies approved evidence inputs; this package does not
own capture, decode, inference, tracking, or vendor SDK objects.

## Clip recording

The smart record actor owns primary clip creation. Clip publication binds the
artifact to its event and sealed sidecar. Decoded frames are analysis and
snapshot taps only. A trigger failure must not block the alert; an incomplete
artifact is recorded through the durable path rather than silently discarded.

## Durable staging and delivery queue

The stager writes durable event/clip work before relay delivery. The delivery
queue is publish-once and owned by one worker process. Retry classes retry,
compatibility failures reprobe, and payload-invalid failures are permanent.
Never use the backend database or a JSON state store for queue state.

## Snapshot store

Snapshots are bounded content-addressed evidence. Keep their lifetime,
retention, and sidecar identity aligned with the published clip. Snapshot work
must not delay alert admission or relay delivery.
