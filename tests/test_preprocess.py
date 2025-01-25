from src.preprocess import load_data

def test_load_data():
    X, y = load_data('data/iris.csv')
    assert X.shape[1] == 4  # 4 features
    assert len(y.unique()) == 3  # 3 classes
