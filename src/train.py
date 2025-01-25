from sklearn.ensemble import RandomForestClassifier
from joblib import dump
from preprocess import load_data, split_data

def train_model(X_train, y_train):
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)
    return model

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    model = train_model(X_train, y_train)
    dump(model, 'models/model.pkl')
    print("Model trained and saved successfully!")
