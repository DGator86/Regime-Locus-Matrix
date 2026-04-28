"""
Massive (Polygon-compatible) **Flat Files** — bulk historical CSV.gz on an S3-compatible endpoint.

REST (`MassiveClient`) is for targeted pulls; Flat Files are for large options backfills
(trades, quotes, day/minute aggregates). See:
https://massive.com/docs/flat-files/quickstart and options dataset pages under ``/flat-files/options/``.

Environment (from Massive dashboard — **not** the same as ``MASSIVE_API_KEY`` alone):

- ``MASSIVE_S3_ENDPOINT`` — S3-compatible endpoint URL (HTTPS).
- ``MASSIVE_S3_ACCESS_KEY`` / ``MASSIVE_S3_SECRET_KEY`` — flat-file credentials.
- ``MASSIVE_S3_BUCKET`` — bucket name (common default: ``flatfiles``; confirm in dashboard).
- Optional: ``MASSIVE_S3_REGION`` — signing region for SigV4 (default ``us-east-1``; use if Massive docs specify another).
- Optional: ``MASSIVE_FLATFILES_PREFIX_TRADES`` etc. to override S3 key prefixes.
- Optional: ``MASSIVE_FLATFILES_KEY_STYLE`` — ``year_month_date`` (default for OPRA layout) or ``year_date``.

Default OPRA-style prefixes match community layout (verify if your keys 404):

- ``us_options_opra/trades_v1``
- ``us_options_opra/quotes_v1``
- ``us_options_opra/day_aggs_v1``
- ``us_options_opra/minute_aggs_v1``
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import date
from typing import Callable, Literal

from dotenv import load_dotenv

FlatfileDataset = Literal["trades", "quotes", "day_aggs", "minute_aggs"]

_DEFAULT_PREFIX: dict[FlatfileDataset, str] = {
    "trades": "us_options_opra/trades_v1",
    "quotes": "us_options_opra/quotes_v1",
    "day_aggs": "us_options_opra/day_aggs_v1",
    "minute_aggs": "us_options_opra/minute_aggs_v1",
}

_ENV_PREFIX_OVERRIDE: dict[FlatfileDataset, str] = {
    "trades": "MASSIVE_FLATFILES_PREFIX_TRADES",
    "quotes": "MASSIVE_FLATFILES_PREFIX_QUOTES",
    "day_aggs": "MASSIVE_FLATFILES_PREFIX_DAY_AGGS",
    "minute_aggs": "MASSIVE_FLATFILES_PREFIX_MINUTE_AGGS",
}


def _key_year_date(prefix: str, d: date) -> str:
    return f"{prefix.rstrip('/')}/{d.year}/{d.isoformat()}.csv.gz"


def _key_year_month_date(prefix: str, d: date) -> str:
    return f"{prefix.rstrip('/')}/{d.year}/{d.month:02d}/{d.isoformat()}.csv.gz"


_KEY_BUILDERS: dict[str, Callable[[str, date], str]] = {
    "year_date": _key_year_date,
    "year_month_date": _key_year_month_date,
}


@dataclass(frozen=True)
class MassiveFlatfilesConfig:
    endpoint_url: str
    access_key: str
    secret_key: str
    bucket: str
    key_style: str = "year_month_date"
    region_name: str = "us-east-1"


def load_massive_flatfiles_config() -> MassiveFlatfilesConfig:
    load_dotenv()
    endpoint = (
        os.environ.get("MASSIVE_S3_ENDPOINT") or os.environ.get("POLYGON_S3_ENDPOINT") or ""
    ).strip()
    if not endpoint:
        raise ValueError(
            "Set MASSIVE_S3_ENDPOINT to the S3-compatible flat-files URL from your Massive dashboard "
            "(Flat Files access is separate from REST apiKey-only usage)."
        )
    key = (
        os.environ.get("MASSIVE_S3_ACCESS_KEY") or os.environ.get("POLYGON_S3_ACCESS_KEY") or ""
    ).strip()
    secret = (
        os.environ.get("MASSIVE_S3_SECRET_KEY") or os.environ.get("POLYGON_S3_SECRET_KEY") or ""
    ).strip()
    if not key or not secret:
        raise ValueError(
            "Set MASSIVE_S3_ACCESS_KEY and MASSIVE_S3_SECRET_KEY for flat-file downloads "
            "(dashboard S3 credentials; not MASSIVE_API_KEY)."
        )
    bucket = (
        os.environ.get("MASSIVE_S3_BUCKET") or os.environ.get("POLYGON_S3_BUCKET") or "flatfiles"
    ).strip()
    # Massive OPRA layout uses ``.../YYYY/MM/YYYY-MM-DD.csv.gz`` (see bucket listing).
    style = (os.environ.get("MASSIVE_FLATFILES_KEY_STYLE") or "year_month_date").strip().lower()
    if style not in _KEY_BUILDERS:
        raise ValueError(
            f"Unknown MASSIVE_FLATFILES_KEY_STYLE={style!r}; use year_date or year_month_date"
        )
    region = (os.environ.get("MASSIVE_S3_REGION") or "us-east-1").strip()
    return MassiveFlatfilesConfig(
        endpoint_url=endpoint.rstrip("/"),
        access_key=key,
        secret_key=secret,
        bucket=bucket,
        key_style=style,
        region_name=region,
    )


def options_flatfile_prefix(dataset: FlatfileDataset) -> str:
    env_name = _ENV_PREFIX_OVERRIDE[dataset]
    override = (os.environ.get(env_name) or "").strip()
    if override:
        return override.rstrip("/")
    return _DEFAULT_PREFIX[dataset]


def options_flatfile_object_key(
    dataset: FlatfileDataset,
    trade_date: date,
    *,
    key_style: str | None = None,
) -> str:
    prefix = options_flatfile_prefix(dataset)
    style = (
        (key_style or os.environ.get("MASSIVE_FLATFILES_KEY_STYLE") or "year_month_date")
        .strip()
        .lower()
    )
    builder = _KEY_BUILDERS.get(style)
    if builder is None:
        raise ValueError(f"Unknown key style {style!r}")
    return builder(prefix, trade_date)


def boto3_s3_client(cfg: MassiveFlatfilesConfig | None = None):
    """Return a boto3 S3 client pointed at the flat-files endpoint (Polygon/Massive-compatible signing)."""
    try:
        import boto3
        from botocore.config import Config
    except ImportError as e:
        raise ImportError("Install boto3: pip install 'regime-locus-matrix[flatfiles]'") from e

    c = cfg or load_massive_flatfiles_config()
    # SigV4 + path-style matches third-party flat-file guides; region is required for signing even
    # though the host is not AWS (see Massive/Polygon flat-file S3 docs and community examples).
    botocore_cfg = Config(
        signature_version="s3v4",
        s3={"addressing_style": "path"},
    )
    session = boto3.session.Session()
    return session.client(
        "s3",
        endpoint_url=c.endpoint_url,
        aws_access_key_id=c.access_key,
        aws_secret_access_key=c.secret_key,
        region_name=c.region_name,
        config=botocore_cfg,
    )


def download_s3_object(
    bucket: str,
    key: str,
    dest_path: str,
    *,
    cfg: MassiveFlatfilesConfig | None = None,
) -> None:
    """Stream object to disk via ``get_object`` (avoids managed-transfer head/stat on some gateways)."""
    import shutil

    client = boto3_s3_client(cfg)
    resp = client.get_object(Bucket=bucket, Key=key)
    body = resp["Body"]
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(body, f, length=8 * 1024 * 1024)
