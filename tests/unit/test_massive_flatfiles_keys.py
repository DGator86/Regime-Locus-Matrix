from datetime import date

import pytest

from rlm.data.massive_flatfiles import options_flatfile_object_key


def test_options_flatfile_object_key_year_date(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MASSIVE_FLATFILES_KEY_STYLE", raising=False)
    monkeypatch.delenv("MASSIVE_FLATFILES_PREFIX_TRADES", raising=False)
    k = options_flatfile_object_key("trades", date(2025, 6, 15), key_style="year_date")
    assert k == "us_options_opra/trades_v1/2025/2025-06-15.csv.gz"


def test_options_flatfile_object_key_year_month_date(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MASSIVE_FLATFILES_PREFIX_QUOTES", raising=False)
    k = options_flatfile_object_key("quotes", date(2025, 6, 15), key_style="year_month_date")
    assert k == "us_options_opra/quotes_v1/2025/06/2025-06-15.csv.gz"


def test_options_flatfile_prefix_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MASSIVE_FLATFILES_PREFIX_DAY_AGGS", "custom/prefix/day")
    k = options_flatfile_object_key("day_aggs", date(2024, 1, 2), key_style="year_date")
    assert k == "custom/prefix/day/2024/2024-01-02.csv.gz"
