from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the locally saved model
model_path = "models/model.pkl"
model = joblib.load(model_path)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Parse the incoming JSON payload
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400

        # Convert the data into a DataFrame
        df = pd.DataFrame(data)
        
        # Make predictions
        predictions = model.predict(df)
        
        # Return the predictions as a JSON response
        return jsonify({"predictions": predictions.tolist()}), 200

    except Exception as e:
        # Handle exceptions and return an error response
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
