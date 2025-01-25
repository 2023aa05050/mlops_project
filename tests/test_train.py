from src.train import train_model
import os

def test_train():
    train_model()
    assert os.path.exists('models/model.pkl')
