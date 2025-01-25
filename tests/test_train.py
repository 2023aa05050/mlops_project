from src.train import train_model
import pandas as pd

def test_train_model():
    X = pd.DataFrame([[1, 2, 3, 4], [5, 6, 7, 8]])
    y = pd.Series([0, 1])
    model = train_model(X, y)
    assert model is not None
