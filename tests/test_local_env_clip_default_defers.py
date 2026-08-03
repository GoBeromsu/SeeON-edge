"""PR #137 (clip always-on) + this branch: silence must defer to
``ClipRecordingConfig``'s own default, not a hardcoded ``False``.

``worker.runtime.config.local_env.clip_recording_config_from_environment``
previously read ``ML_WORKER_CLIP_RECORDING_ENABLED`` via a boolean parser
that collapsed "unset" to an explicit ``False`` before it ever reached
``ClipRecordingConfig``, so any future change to that model's own
``enabled`` default (e.g. flipping it to always-on) would be silently
overridden on every unconfigured boot. This pins "explicit wins outright,
silence defers" for the env layer itself: on this branch (based on
``main``) ``ClipRecordingConfig.enabled`` still defaults to ``False``, so
the silence case resolves to ``False`` here -- once the always-on default
change (PR #137) merges, this same assertion flips to ``True`` with no
further code change required, because the resolution now genuinely defers
to the model default instead of re-asserting ``False`` itself.
"""

from __future__ import annotations

import pytest

from worker.runtime.config.local_env import (
    ML_WORKER_CLIP_RECORDING_ENABLED_ENV,
    clip_recording_config_from_environment,
)
from worker.runtime.config.worker_models import ClipRecordingConfig


def test_silent_env_defers_to_clip_recording_config_model_default() -> None:
    config = clip_recording_config_from_environment({})

    assert config.enabled == ClipRecordingConfig().enabled


@pytest.mark.parametrize("raw,expected", [("true", True), ("false", False)])
def test_explicit_env_value_wins_outright_over_model_default(
    raw: str, expected: bool
) -> None:
    config = clip_recording_config_from_environment(
        {ML_WORKER_CLIP_RECORDING_ENABLED_ENV: raw}
    )

    assert config.enabled is expected
