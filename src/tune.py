import optuna
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from src.preprocess import load_data

# Objective function for Optuna optimization
def objective(trial):
    X, y = load_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define hyperparameters to tune
    n_estimators = trial.suggest_int('n_estimators', 50, 200)
    max_depth = trial.suggest_int('max_depth', 5, 20)
    
    # Train the model
    model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    
    return accuracy

def tune_model():
    # Optuna optimization process
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    
    print(f"Best trial: {study.best_trial.value}")
    print(f"Best parameters: {study.best_trial.params}")

if __name__ == "__main__":
    tune_model()
