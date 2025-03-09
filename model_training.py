import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib

# Load preprocessed data
X_train = np.load("epigenome/X_train.npy")
X_test = np.load("epigenome/X_test.npy")
y_train = np.load("epigenome/y_train.npy")
y_test = np.load("epigenome/y_test.npy")

# Define models
models = {
    "RandomForest": RandomForestRegressor(n_estimators=100, random_state=42),
    "XGBoost": XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=42),
    "NeuralNetwork": MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=500, random_state=42)
}

# Train and evaluate models
results = {}
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Compute metrics
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    results[name] = {"MAE": mae, "RMSE": rmse}
    
    # Save trained model
    joblib.dump(model, f"epigenome/{name}_model.pkl")
    print(f"{name} - MAE: {mae:.4f}, RMSE: {rmse:.4f}")

# Display results
results_df = pd.DataFrame(results).T
print("\nModel Performance:")
print(results_df)
