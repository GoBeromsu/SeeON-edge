# worker/interfaces: worker-internal seams

One Protocol per seam. Interfaces contain no vendor imports, business policy,
or runtime composition. They depend only on `worker.types`; implementations
live in adapters or pipeline and are constructed by `worker.runtime`.

## Ownership

- `media_plane.py`: Flow media-plane lifecycle and callbacks. SDK objects never
  cross this boundary.
- `association.py`: association inputs and outputs used by policy.
- `perception.py`: normalized post-Flow observations.
- `serving.py`: CPU model-serving seam for domain policy helpers.
- `output.py`: event and evidence publication seam.
- `source_packet.py`: source-packet identity used by clip evidence.
- `frame.py`: host-frame materialization boundary.

A new seam is a Protocol plus two implementations, or one implementation plus
a test double. Do not let a vendor type, a config resolver, or a runtime object
leak through a signature. A caller owns input lifetime until transfer is
explicit; an implementation owns resources it allocates and closes them on its
lifecycle path.

Focused tests: `tests/test_worker_interfaces.py`,
`tests/test_deepstream_adapter_plane.py`. Boundary:
`uv run --group lint lint-imports`.
