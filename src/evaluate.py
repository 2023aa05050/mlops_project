from sklearn.metrics import classification_report
from joblib import load
from preprocess import load_data, split_data

def evaluate_model(X_test, y_test, model_path='models/model.pkl'):
    model = load(model_path)
    predictions = model.predict(X_test)
    report = classification_report(y_test, predictions)
    return report

if __name__ == "__main__":
    X, y = load_data()
    _, X_test, _, y_test = split_data(X, y)
    report = evaluate_model(X_test, y_test)
    print("Model Evaluation Report:")
    print(report)
