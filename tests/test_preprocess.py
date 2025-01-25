from src.preprocess import preprocess

def test_preprocess():
    X_train, X_test, y_train, y_test = preprocess('data/iris.csv')
    assert len(X_train) > 0
    assert len(X_test) > 0
