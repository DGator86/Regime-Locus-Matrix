"""Three-track trading configuration (equities, large options, SPY day trade)."""

from rlm.trading.tracks import (
    TRACK_LARGE_EQUITIES,
    TRACK_LARGE_OPTIONS,
    TRACK_SPY_DAYTRADE,
    TrackSpec,
    load_tracks,
    print_tracks_banner,
    resolve_root,
)

__all__ = [
    "TRACK_LARGE_EQUITIES",
    "TRACK_LARGE_OPTIONS",
    "TRACK_SPY_DAYTRADE",
    "TrackSpec",
    "load_tracks",
    "print_tracks_banner",
    "resolve_root",
]
