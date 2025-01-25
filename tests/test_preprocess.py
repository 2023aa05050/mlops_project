from src.preprocess import load_data, split_data

def test_preprocess():
    X, y = load_data('data/iris.csv')
    X_train, X_test, y_train, y_test = split_data(X, y)
    
    assert len(X_train) > 0
    assert len(X_test) > 0
