import numpy as np
import joblib

def load_models():
    models = {
        "RandomForest": joblib.load("epigenome/RandomForest_model.pkl"),
        "XGBoost": joblib.load("epigenome/XGBoost_model.pkl"),
        "NeuralNetwork": joblib.load("epigenome/NeuralNetwork_model.pkl")
    }
    return models

def predict_plant_traits(input_data, model_name="RandomForest"):
    models = load_models()
    model = models.get(model_name)
    if model is None:
        raise ValueError(f"Model {model_name} not found.")
    
    input_array = np.array(input_data).reshape(1, -1)
    prediction = model.predict(input_array)
    return prediction.flatten()

if __name__ == "__main__":
    # Example user input (ensure categorical variables are properly encoded before prediction)
    sample_input = [13.85, 26.34, 49.51, 58.42, 64.87, 57.60, 46.75, 43.32, 12.75, 39.01, 18, 45.17]
    
    predicted_traits = predict_plant_traits(sample_input, model_name="RandomForest")
    print("Predicted Traits:", predicted_traits)
