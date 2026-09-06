# worker/adapters/deepstream

This is the sole worker package allowed to import DeepStream vendor modules.
Keep those imports lazy so host tests import this package without an NVIDIA
runtime. Convert vendor metadata immediately into worker envelopes; callers use
only `worker.interfaces` and `worker.types`.
