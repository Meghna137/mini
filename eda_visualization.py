import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("epigenome/methylation_dataset.csv")

# Set the style
sns.set_style("whitegrid")

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap of Features and Target Variables")
plt.show()

# Boxplot: Promoter Methylation across Plant Species
plt.figure(figsize=(12, 6))
sns.boxplot(x="Plant Species", y="Promoter Methylation (%)", data=df)
plt.title("Distribution of Promoter Methylation Across Plant Species")
plt.xticks(rotation=45)
plt.show()

# Scatter Plot: Yield vs. Promoter Methylation
plt.figure(figsize=(8, 6))
sns.scatterplot(x=df["Promoter Methylation (%)"], y=df["Yield (g/plant)"], hue=df["Plant Species"], alpha=0.7)
plt.title("Yield vs. Promoter Methylation")
plt.xlabel("Promoter Methylation (%)")
plt.ylabel("Yield (g/plant)")
plt.show()

# Pairplot for key features and target traits
selected_features = ["Promoter Methylation (%)", "CHH Methylation (%)", "Soil Moisture (%)", "Yield (g/plant)"]
sns.pairplot(df[selected_features + ["Plant Species"]], hue="Plant Species")
plt.show()

print("EDA visualizations complete!")
