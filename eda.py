import os
import pandas as pd
import matplotlib.pyplot as plt

print("=" * 50)
print("       SALES PREDICTION - EDA")
print("=" * 50)

# Get current project directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "sales.csv")
GRAPH_DIR = os.path.join(BASE_DIR, "graphs")

# Create graphs folder if it doesn't exist
os.makedirs(GRAPH_DIR, exist_ok=True)

# Check if dataset exists
if not os.path.exists(DATASET_PATH):
    print(f"\n❌ Dataset not found!")
    print(f"Expected location:\n{DATASET_PATH}")
    exit()

# Load dataset
df = pd.read_csv(DATASET_PATH)

print("\n✅ Dataset Loaded Successfully!")
print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Shape:")
print(df.shape)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nMissing Values:")
print(df.isnull().sum())

print("\nStatistical Summary:")
print(df.describe())

# -----------------------------
# 1. Sales Distribution
# -----------------------------
plt.figure(figsize=(8, 5))
plt.hist(df["Sales"], bins=20)
plt.title("Sales Distribution")
plt.xlabel("Sales")
plt.ylabel("Frequency")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "sales_distribution.png"))
plt.close()

# -----------------------------
# 2. Advertising vs Sales
# -----------------------------
plt.figure(figsize=(8, 5))
plt.scatter(df["Advertising"], df["Sales"])
plt.title("Advertising vs Sales")
plt.xlabel("Advertising Budget")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "advertising_vs_sales.png"))
plt.close()

# -----------------------------
# 3. Price vs Sales
# -----------------------------
plt.figure(figsize=(8, 5))
plt.scatter(df["Price"], df["Sales"])
plt.title("Price vs Sales")
plt.xlabel("Price")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "price_vs_sales.png"))
plt.close()

# -----------------------------
# 4. Discount vs Sales
# -----------------------------
plt.figure(figsize=(8, 5))
plt.scatter(df["Discount"], df["Sales"])
plt.title("Discount vs Sales")
plt.xlabel("Discount (%)")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "discount_vs_sales.png"))
plt.close()

# -----------------------------
# 5. Season vs Sales
# -----------------------------
plt.figure(figsize=(8, 5))
plt.scatter(df["Season"], df["Sales"])
plt.title("Season vs Sales")
plt.xlabel("Season")
plt.ylabel("Sales")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "season_vs_sales.png"))
plt.close()

# -----------------------------
# 6. Correlation Matrix
# -----------------------------
corr = df.corr(numeric_only=True)

plt.figure(figsize=(6, 6))
plt.imshow(corr, cmap="coolwarm")
plt.colorbar()

plt.xticks(range(len(corr.columns)), corr.columns, rotation=45)
plt.yticks(range(len(corr.columns)), corr.columns)

for i in range(len(corr.columns)):
    for j in range(len(corr.columns)):
        plt.text(j, i, f"{corr.iloc[i, j]:.2f}",
                 ha="center", va="center", fontsize=8)

plt.title("Correlation Matrix")
plt.tight_layout()
plt.savefig(os.path.join(GRAPH_DIR, "correlation_matrix.png"))
plt.close()

print("\n" + "=" * 50)
print("✅ EDA COMPLETED SUCCESSFULLY!")
print("=" * 50)
print(f"\n📂 Graphs saved in:\n{GRAPH_DIR}")

print("\nGenerated Graphs:")
print("✔ sales_distribution.png")
print("✔ advertising_vs_sales.png")
print("✔ price_vs_sales.png")
print("✔ discount_vs_sales.png")
print("✔ season_vs_sales.png")
print("✔ correlation_matrix.png")