from flask import Flask, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load trained models
models = {
    "RandomForest": joblib.load("epigenome/RandomForest_model.pkl"),
    "XGBoost": joblib.load("epigenome/XGBoost_model.pkl"),
    "NeuralNetwork": joblib.load("epigenome/NeuralNetwork_model.pkl")
}

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    model_name = data.get("model", "RandomForest")  # Default to RandomForest
    
    # Extract input features
    features = np.array(data["features"]).reshape(1, -1)
    
    # Get the model
    model = models.get(model_name)
    if model is None:
        return jsonify({"error": f"Model {model_name} not found"}), 400
    
    # Make prediction
    prediction = model.predict(features).flatten().tolist()
    
    return jsonify({"model": model_name, "predicted_traits": prediction})

if __name__ == "__main__":
    app.run(debug=True)
