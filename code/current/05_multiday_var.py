import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

# 05_multiday_var.py

BASE_PATH = "/Users/aaronroman/Desktop/QFRM"
INPUT_PATH = f"{BASE_PATH}/data/final_analysis_2026_04_09"
RESULTS_PATH = f"{BASE_PATH}/results/final_analysis_2026_04_09"
FIGURES_PATH = f"{BASE_PATH}/figures/final_analysis_2026_04_09"

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

# Load rebuilt portfolio dataset
df = pd.read_csv(f"{INPUT_PATH}/01_portfolio_with_returns_and_losses.csv")

# Settings
alpha = 0.99
V = 1_000_000

# Daily portfolio return series
r1 = df["portfolio_return"].dropna().reset_index(drop=True)

def non_overlapping_returns(series, horizon):
    n = len(series)
    usable = (n // horizon) * horizon
    trimmed = series.iloc[:usable].to_numpy()
    grouped = trimmed.reshape(-1, horizon)
    # compound returns over each non-overlapping block
    compounded = np.prod(1 + grouped, axis=1) - 1
    return pd.Series(compounded)

def historical_var_es(return_series, alpha, V):
    losses_ret = -return_series
    losses_eur = V * losses_ret

    var_ret = np.quantile(losses_ret, alpha)
    es_ret = losses_ret[losses_ret >= var_ret].mean()

    var_eur = np.quantile(losses_eur, alpha)
    es_eur = losses_eur[losses_eur >= var_eur].mean()

    return var_ret, es_ret, var_eur, es_eur

# 1-day historical VaR/ES
var_1d_ret, es_1d_ret, var_1d_eur, es_1d_eur = historical_var_es(r1, alpha, V)

# 5-day non-overlapping
r5 = non_overlapping_returns(r1, 5)
var_5d_ret, es_5d_ret, var_5d_eur, es_5d_eur = historical_var_es(r5, alpha, V)

# 10-day non-overlapping
r10 = non_overlapping_returns(r1, 10)
var_10d_ret, es_10d_ret, var_10d_eur, es_10d_eur = historical_var_es(r10, alpha, V)

# Square-root-of-time approximation from 1-day VaR
var_5d_sqrt_ret = np.sqrt(5) * var_1d_ret
var_10d_sqrt_ret = np.sqrt(10) * var_1d_ret

var_5d_sqrt_eur = np.sqrt(5) * var_1d_eur
var_10d_sqrt_eur = np.sqrt(10) * var_1d_eur

# Results table
results = pd.DataFrame({
    "Horizon": ["1-day", "5-day", "10-day"],
    "Historical_VaR_return": [var_1d_ret, var_5d_ret, var_10d_ret],
    "Historical_ES_return": [es_1d_ret, es_5d_ret, es_10d_ret],
    "Historical_VaR_EUR": [var_1d_eur, var_5d_eur, var_10d_eur],
    "Historical_ES_EUR": [es_1d_eur, es_5d_eur, es_10d_eur],
    "SqrtTime_VaR_return": [var_1d_ret, var_5d_sqrt_ret, var_10d_sqrt_ret],
    "SqrtTime_VaR_EUR": [var_1d_eur, var_5d_sqrt_eur, var_10d_sqrt_eur]
})

print(results)

results.to_csv(f"{RESULTS_PATH}/05_multiday_var_results.csv", index=False)

# Plot comparison in EUR
plot_df = pd.DataFrame({
    "Horizon": ["1-day", "5-day", "10-day"],
    "Historical VaR": [var_1d_eur, var_5d_eur, var_10d_eur],
    "Sqrt-time VaR": [var_1d_eur, var_5d_sqrt_eur, var_10d_sqrt_eur]
})

plot_df.set_index("Horizon").plot(figsize=(8, 5))
plt.title("Historical Multi-day VaR vs Square-root-of-time VaR")
plt.ylabel("VaR (EUR)")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/05_multiday_var_comparison.png", dpi=300)
plt.show()

print("\nSaved files:")
print(f"{RESULTS_PATH}/05_multiday_var_results.csv")
print(f"{FIGURES_PATH}/05_multiday_var_comparison.png")