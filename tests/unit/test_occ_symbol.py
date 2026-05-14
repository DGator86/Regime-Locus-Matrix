from datetime import date

from rlm.data.occ_symbol import format_occ_compact_symbol, parse_occ_option_symbol


def test_format_occ_round_trips_with_parser() -> None:
    d = date(2026, 2, 13)
    s = format_occ_compact_symbol("SPY", d, option_type="call", strike=450.0)
    assert s == "SPY260213C00450000"
    p = parse_occ_option_symbol(s)
    assert p.root == "SPY"
    assert p.strike == 450.0
    assert p.option_type == "call"


def test_parse_occ_with_o_prefix_matches_plain() -> None:
    a = parse_occ_option_symbol("AAPL240202P00185000")
    b = parse_occ_option_symbol("O:AAPL240202P00185000")
    assert a.root == b.root == "AAPL"
    assert a.strike == b.strike == 185.0
    assert a.option_type == b.option_type == "put"
