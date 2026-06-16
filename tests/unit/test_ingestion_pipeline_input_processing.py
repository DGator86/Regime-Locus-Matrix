import pytest

from rlm.ingestion.pipeline import _csv_list, _positive_int, build_parser


def test_csv_list_trims_and_filters_empty_items() -> None:
    assert _csv_list(" SPY, ,qqq ,, IWM ", uppercase=True) == ["SPY", "QQQ", "IWM"]
    assert _csv_list(" A , b , , c ") == ["A", "b", "c"]


def test_positive_int_rejects_non_positive_values() -> None:
    assert _positive_int("3") == 3
    with pytest.raises(Exception):
        _positive_int("0")
    with pytest.raises(Exception):
        _positive_int("-2")


def test_ingestion_parser_validates_jobs_flag() -> None:
    parser = build_parser()
    args = parser.parse_args(["--jobs", "2"])
    assert args.jobs == 2

    with pytest.raises(SystemExit):
        parser.parse_args(["--jobs", "0"])
