import pandas as pd
import os

# 01_build_portfolio_with_returns_and_losses.py

BASE_PATH = "/Users/aaronroman/Desktop/QFRM"
DATA_PATH = f"{BASE_PATH}/data/raw"
OUTPUT_PATH = f"{BASE_PATH}/data/final_analysis_2026_04_09"

os.makedirs(OUTPUT_PATH, exist_ok=True)


# Load final portfolio return dataset
df = pd.read_csv(f"{DATA_PATH}/portfolio_returns_clean.csv")

# Fixed portfolio weights
weights = {
    "MSFT": 0.20,
    "SHEL": 0.15,
    "JPM": 0.15,
    "^GSPC": 0.20,
    "EURUSD=X": 0.10,
    "LOAN": 0.20
}

# Portfolio value
V = 1_000_000

# Build weighted return columns
df["w_MSFT"] = weights["MSFT"] * df["MSFT"]
df["w_SHEL"] = weights["SHEL"] * df["SHEL"]
df["w_JPM"] = weights["JPM"] * df["JPM"]
df["w_^GSPC"] = weights["^GSPC"] * df["^GSPC"]
df["w_EURUSD=X"] = weights["EURUSD=X"] * df["EURUSD=X"]
df["w_LOAN"] = weights["LOAN"] * df["LOAN"]

# Build portfolio return
df["portfolio_return"] = (
    df["w_MSFT"]
    + df["w_SHEL"]
    + df["w_JPM"]
    + df["w_^GSPC"]
    + df["w_EURUSD=X"]
    + df["w_LOAN"]
)

# Build portfolio loss
df["portfolio_loss_pct"] = -df["portfolio_return"]
df["portfolio_loss_eur"] = -V * df["portfolio_return"]

# Save rebuilt final dataset
df.to_csv(f"{OUTPUT_PATH}/01_portfolio_with_returns_and_losses.csv", index=False)

print("Rebuilt file saved:")
print(f"{OUTPUT_PATH}/01_portfolio_with_returns_and_losses.csv")
print(df[[
    "Date", "portfolio_return", "portfolio_loss_pct", "portfolio_loss_eur"
]].head())