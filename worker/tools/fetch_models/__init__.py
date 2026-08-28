"""Model provisioning for the worker's ``/app/models`` volume.

One-shot tool run by the ``edge-model-fetch`` compose service (and by
``scripts/fetch-models.sh`` on dev hosts). It downloads every artifact listed
in the committed ``manifest.json`` from its pinned upstream (a Hugging Face
revision or a GitHub release tag), verifies size and SHA-256, and writes the
git-tracked LSTM sidecars beside the weights. Stdlib only; never imported by
``python -m worker``.
"""
