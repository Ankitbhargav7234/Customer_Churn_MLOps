import os

def test_preprocessor_exists():
    assert os.path.exists("artifacts/preprocessor.pkl")