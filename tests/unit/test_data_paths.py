from rlm.data.paths import get_data_root


def test_data_root_override(tmp_path):
    assert get_data_root(tmp_path) == tmp_path.resolve()
