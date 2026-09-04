"""DeepStream media plane (P1b, G8b).

The only package in the worker that may import ``pyservicemaker`` or ``pyds``
(import-linter: "only worker.adapters.deepstream imports pyservicemaker or
pyds"). It implements ``worker.interfaces.media_plane.MediaPlane`` over a
pyservicemaker Flow and converts ``NvDsBatchMeta`` into ``PerceptionFrameV1``
metadata at the boundary; nothing DeepStream-shaped leaves this package.
"""
