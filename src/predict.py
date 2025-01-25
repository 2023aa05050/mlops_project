from joblib import load

def predict(input_data, model_path='models/model.pkl'):
    model = load(model_path)
    return model.predict([input_data])

if __name__ == "__main__":
    sample = [5.1, 3.5, 1.4, 0.2]  # Example input
    prediction = predict(sample)
    print(f"Predicted Class: {prediction[0]}")
