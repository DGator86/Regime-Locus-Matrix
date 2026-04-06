from rlm.data.occ_symbol import parse_occ_option_symbol


def test_parse_occ_with_o_prefix_matches_plain() -> None:
    a = parse_occ_option_symbol("AAPL240202P00185000")
    b = parse_occ_option_symbol("O:AAPL240202P00185000")
    assert a.root == b.root == "AAPL"
    assert a.strike == b.strike == 185.0
    assert a.option_type == b.option_type == "put"
