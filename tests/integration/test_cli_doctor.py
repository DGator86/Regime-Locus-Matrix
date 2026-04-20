from rlm.core.services.diagnostics_service import DiagnosticsService


def test_doctor_json_payload(tmp_path):
    report = DiagnosticsService().run(data_root=str(tmp_path), backend="auto", symbol="SPY")
    payload = report.to_dict()
    assert "checks" in payload
