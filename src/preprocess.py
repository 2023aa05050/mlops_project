import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path='data/Iris.csv'):
    data = pd.read_csv(file_path)
    X = data.drop(['species', 'Id'], axis=1)  # Exclude 'Id' from features
    y = data['species']
    return X, y


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)

if __name__ == "__main__":
    X, y = load_data()
    X_train, X_test, y_train, y_test = split_data(X, y)
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
