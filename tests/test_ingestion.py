import os

def test_data_ingestion():
    assert os.path.exists("artifacts/train.csv")
    assert os.path.exists("artifacts/test.csv")