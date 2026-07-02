import pandas as pd
import numpy as np

# For reproducibility
np.random.seed(42)

rows = 1000

advertising = np.random.randint(10, 100, rows)      # Advertising budget (₹1000s)
price = np.random.randint(50, 500, rows)            # Product price (₹)
discount = np.random.randint(0, 40, rows)           # Discount (%)
season = np.random.randint(1, 5, rows)              # 1=Winter,2=Summer,3=Festival,4=Regular

# Generate realistic sales
sales = (
    advertising * 8
    - price * 1.5
    + discount * 12
    + season * 40
    + np.random.normal(0, 50, rows)
)

sales = sales.astype(int)

sales[sales < 20] = 20

df = pd.DataFrame({
    "Advertising": advertising,
    "Price": price,
    "Discount": discount,
    "Season": season,
    "Sales": sales
})

df.to_csv("dataset/sales.csv", index=False)

print("Dataset created successfully!")
print(df.head())