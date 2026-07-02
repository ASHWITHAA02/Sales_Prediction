import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

print("=" * 60)
print("BUSINESS SALES PREDICTION - MODEL TRAINING")
print("=" * 60)

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "sales.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")

os.makedirs(MODEL_DIR, exist_ok=True)

# -------------------------
# Load Dataset
# -------------------------
df = pd.read_csv(DATASET_PATH)

print("\nDataset Loaded Successfully!")
print("Dataset Shape:", df.shape)

# -------------------------
# Features and Target
# -------------------------
X = df[["Advertising", "Price", "Discount", "Season"]]
y = df["Sales"]

# -------------------------
# Split Dataset
# -------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print(f"\nTraining Samples : {len(X_train)}")
print(f"Testing Samples  : {len(X_test)}")

# -------------------------
# Train Model
# -------------------------
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

print("\nModel Training Completed!")

# -------------------------
# Prediction
# -------------------------
y_pred = model.predict(X_test)

# -------------------------
# Evaluation
# -------------------------
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
r2 = r2_score(y_test, y_pred)

print("\nModel Performance")
print("-" * 30)
print(f"MAE  : {mae:.2f}")
print(f"MSE  : {mse:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R² Score : {r2:.4f}")

# -------------------------
# Save Model
# -------------------------
MODEL_PATH = os.path.join(MODEL_DIR, "sales_prediction_model.pkl")

joblib.dump(model, MODEL_PATH)

print("\nModel Saved Successfully!")
print(MODEL_PATH)

# -------------------------
# Feature Importance
# -------------------------
importance = model.feature_importances_

print("\nFeature Importance")
print("-" * 30)

for feature, score in zip(X.columns, importance):
    print(f"{feature:15} : {score:.4f}")

print("\nProject Training Completed Successfully!")