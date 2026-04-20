"""Lake metadata — tracks what symbols/intervals are present in the lake."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from rlm.data.paths import get_data_root
from rlm.utils.logging import get_logger

log = get_logger(__name__)

_METADATA_FILENAME = "metadata.json"


@dataclass
class SymbolMeta:
    symbol: str
    intervals: list[str] = field(default_factory=list)
    chain_dates: list[str] = field(default_factory=list)
    last_updated: str = ""


@dataclass
class LakeMetadata:
    symbols: dict[str, SymbolMeta] = field(default_factory=dict)
    schema_version: str = "1"

    @classmethod
    def load(cls, data_root: str | Path | None = None) -> "LakeMetadata":
        """Load metadata from the lake root, or return an empty instance."""
        path = _metadata_path(data_root)
        if not path.is_file():
            return cls()
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            symbols = {
                k: SymbolMeta(**v) for k, v in raw.get("symbols", {}).items()
            }
            return cls(symbols=symbols, schema_version=raw.get("schema_version", "1"))
        except Exception as exc:
            log.warning("lake metadata load failed: %s", exc)
            return cls()

    def save(self, data_root: str | Path | None = None) -> Path:
        """Persist metadata to the lake root."""
        path = _metadata_path(data_root)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {"schema_version": self.schema_version, "symbols": {
            k: asdict(v) for k, v in self.symbols.items()
        }}
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        log.debug("lake metadata saved  path=%s", path)
        return path

    def record_bars(self, symbol: str, interval: str, timestamp: str = "") -> None:
        sym = symbol.upper()
        if sym not in self.symbols:
            self.symbols[sym] = SymbolMeta(symbol=sym)
        if interval not in self.symbols[sym].intervals:
            self.symbols[sym].intervals.append(interval)
        if timestamp:
            self.symbols[sym].last_updated = timestamp

    def record_chain(self, symbol: str, as_of: str) -> None:
        sym = symbol.upper()
        if sym not in self.symbols:
            self.symbols[sym] = SymbolMeta(symbol=sym)
        if as_of not in self.symbols[sym].chain_dates:
            self.symbols[sym].chain_dates.append(as_of)

    def has_bars(self, symbol: str, interval: str) -> bool:
        meta = self.symbols.get(symbol.upper())
        return meta is not None and interval in meta.intervals

    def list_symbols(self) -> list[str]:
        return list(self.symbols.keys())


def _metadata_path(data_root: str | Path | None) -> Path:
    return get_data_root(data_root) / "lake" / _METADATA_FILENAME
