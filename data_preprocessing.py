import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load the dataset
df = pd.read_csv("epigenome/methylation_dataset.csv")  # Ensure the dataset is placed correctly

# Define numerical and categorical columns
numerical_cols = [
    "Soil Moisture (%)", "Temperature (°C)", "Promoter Methylation (%)", "Gene Body Methylation (%)", 
    "TE Methylation (%)", "Global DNA Methylation (%)", "CG Methylation (%)", "CHG Methylation (%)", 
    "CHH Methylation (%)", "siRNA Expression (FPKM)", "SNP Variants", "Gene Expression (FPKM)"
]
categorical_cols = ["Plant Species", "Gene ID", "Developmental Stage", "Stress Treatment", 
                    "Methylation Pattern Type", "Histone Modification"]

# Define target variables (traits to predict)
target_cols = [
    "Drought Resistance (1-10)", "Root Depth (cm)", "Leaf Area Index", 
    "Photosynthetic Rate (µmol CO₂/m²/s)", "Water Use Efficiency", "Chlorophyll Content", "Yield (g/plant)"
]

# Splitting features (X) and target variables (y)
X = df[numerical_cols + categorical_cols]
y = df[target_cols]

# Data Preprocessing Pipeline
preprocessor = ColumnTransformer([
    ("num", MinMaxScaler(), numerical_cols),  # Normalize numerical data
    ("cat", OneHotEncoder(handle_unknown='ignore'), categorical_cols)  # Encode categorical variables
])

# Split the data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Fit the transformer and transform the data
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Save preprocessed data
np.save("epigenome/X_train.npy", X_train)
np.save("epigenome/X_test.npy", X_test)
np.save("epigenome/y_train.npy", y_train)
np.save("epigenome/y_test.npy", y_test)

print("Data preprocessing complete. Train/Test data saved in epigenome folder.")
