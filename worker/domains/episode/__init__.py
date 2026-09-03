"""Shared, per-camera episode lifecycle authority.

Domains score observations and submit proposals here; this package is the only
place that turns an onset into a BusinessEvent or mints its identity.
"""

from worker.domains.episode.authority import EpisodeAuthority, EpisodeProposal, EpisodeState

__all__ = ["EpisodeAuthority", "EpisodeProposal", "EpisodeState"]
