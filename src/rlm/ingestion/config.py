"""Ingestion-specific runtime settings."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IngestionConfig:
    jobs: int = 1
    parallel_backend: str = "process"
    stock_1d_duration: str = "2 Y"
    stock_1d_slug: str = "2y"
    stock_1m_duration: str = "10 D"
    stock_1m_slug: str = "10d"
