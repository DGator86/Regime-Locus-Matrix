from rlm.data.backend import DataBackend


def test_backend_coerce():
    assert DataBackend.coerce("auto") == DataBackend.AUTO
    assert DataBackend.coerce("csv") == DataBackend.CSV
    assert DataBackend.coerce("lake") == DataBackend.LAKE
