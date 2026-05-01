import os
def test_validation_report_exists():
    assert os.path.exists("artifacts/validation_report.json")