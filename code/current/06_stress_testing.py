import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 06_stress_testing.py

BASE_PATH = "/Users/aaronroman/Desktop/QFRM"
INPUT_PATH = f"{BASE_PATH}/data/final_analysis_2026_04_09"
RESULTS_PATH = f"{BASE_PATH}/results/final_analysis_2026_04_09"
FIGURES_PATH = f"{BASE_PATH}/figures/final_analysis_2026_04_09"

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

# Load rebuilt portfolio dataset
df = pd.read_csv(f"{INPUT_PATH}/01_portfolio_with_returns_and_losses.csv")

# Portfolio weights
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

# Base 1-day historical VaR from current portfolio losses
portfolio_losses = df["portfolio_loss_eur"].dropna()
base_var_99 = np.quantile(portfolio_losses, 0.99)

# Stress scenarios
scenarios = {
    "Equity_-20%": {"MSFT": -0.20, "SHEL": -0.20, "JPM": -0.20, "^GSPC": -0.20},
    "Equity_+20%": {"MSFT":  0.20, "SHEL":  0.20, "JPM":  0.20, "^GSPC":  0.20},
    "Equity_-40%": {"MSFT": -0.40, "SHEL": -0.40, "JPM": -0.40, "^GSPC": -0.40},
    "Equity_+40%": {"MSFT":  0.40, "SHEL":  0.40, "JPM":  0.40, "^GSPC":  0.40},

    "FX_-10%": {"EURUSD=X": -0.10},
    "FX_+10%": {"EURUSD=X":  0.10},

    "Rate_+2%": {"LOAN": -0.02},
    "Rate_-2%": {"LOAN":  0.02},
    "Rate_+3%": {"LOAN": -0.03},
    "Rate_-3%": {"LOAN":  0.03},
}

# Baseline portfolio return series
asset_cols = ["MSFT", "SHEL", "JPM", "^GSPC", "EURUSD=X", "LOAN"]

df["baseline_portfolio_return"] = sum(weights[col] * df[col] for col in asset_cols)
df["baseline_portfolio_loss_eur"] = -V * df["baseline_portfolio_return"]

stress_results = []

for scen_name, shocks in scenarios.items():
    stressed_return = df["baseline_portfolio_return"].copy()

    # Apply one-off deterministic shock on top of return series
    shock_component = pd.Series(0.0, index=df.index)

    for asset, shock in shocks.items():
        shock_component += weights[asset] * shock

    stressed_return = stressed_return + shock_component
    stressed_loss_eur = -V * stressed_return

    stressed_var_99 = np.quantile(stressed_loss_eur, 0.99)
    avg_loss = stressed_loss_eur.mean()
    max_loss = stressed_loss_eur.max()

    stress_results.append({
        "Scenario": scen_name,
        "Base_VaR_99_EUR": base_var_99,
        "Stressed_VaR_99_EUR": stressed_var_99,
        "Change_in_VaR_EUR": stressed_var_99 - base_var_99,
        "Average_Loss_EUR": avg_loss,
        "Max_Loss_EUR": max_loss
    })

stress_df = pd.DataFrame(stress_results)
stress_df = stress_df.sort_values("Change_in_VaR_EUR", ascending=False)

print(stress_df)

# Save results
stress_df.to_csv(f"{RESULTS_PATH}/06_stress_testing_results.csv", index=False)

# Plot change in VaR
plt.figure(figsize=(10, 6))
plt.bar(stress_df["Scenario"], stress_df["Change_in_VaR_EUR"])
plt.title("Stress Testing: Change in 99% VaR by Scenario")
plt.xlabel("Scenario")
plt.ylabel("Change in VaR (EUR)")
plt.xticks(rotation=45)
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/06_stress_testing_var_changes.png", dpi=300)
plt.show()

print("\nSaved files:")
print(f"{RESULTS_PATH}/06_stress_testing_results.csv")
print(f"{FIGURES_PATH}/06_stress_testing_var_changes.png")