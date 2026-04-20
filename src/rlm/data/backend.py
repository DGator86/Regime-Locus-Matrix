from __future__ import annotations

from enum import Enum


class DataBackend(str, Enum):
    AUTO = "auto"
    CSV = "csv"
    LAKE = "lake"

    @classmethod
    def coerce(cls, value: str | "DataBackend" | None) -> "DataBackend":
        if isinstance(value, cls):
            return value
        if value is None:
            return cls.AUTO
        return cls(str(value).lower())
