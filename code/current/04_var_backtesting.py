import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import binom

# 04_var_backtesting.py

BASE_PATH = "/Users/aaronroman/Desktop/QFRM"
INPUT_PATH = f"{BASE_PATH}/data/final_analysis_2026_04_09"
RESULTS_PATH = f"{BASE_PATH}/results/final_analysis_2026_04_09"
FIGURES_PATH = f"{BASE_PATH}/figures/final_analysis_2026_04_09"

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

# Load data
df = pd.read_csv(f"{INPUT_PATH}/01_portfolio_with_returns_and_losses.csv")
results = pd.read_csv(f"{RESULTS_PATH}/02_portfolio_var_es_results.csv")

# Settings
alpha = 0.99
alpha_test = 0.05

# Loss series
df["Date"] = pd.to_datetime(df["Date"])
losses = df["portfolio_loss_eur"].dropna().reset_index(drop=True)

# Extract VaR thresholds
var_normal = results.loc[results["Method"] == "Normal", "VaR_EUR"].values[0]
var_t = results.loc[results["Method"] == "Student-t (df=5)", "VaR_EUR"].values[0]
var_hist = results.loc[results["Method"] == "Historical", "VaR_EUR"].values[0]

# Violation indicators
viol_normal = (losses > var_normal).astype(int)
viol_t = (losses > var_t).astype(int)
viol_hist = (losses > var_hist).astype(int)

# Count violations
T = len(losses)
expected_viol = T * (1 - alpha)

count_normal = int(viol_normal.sum())
count_t = int(viol_t.sum())
count_hist = int(viol_hist.sum())

# Binomial critical values
cL = binom.ppf(alpha_test / 2, T, 1 - alpha)
cU = binom.ppf(1 - alpha_test / 2, T, 1 - alpha)

def two_sided_binom_pvalue(count, T, p):
    p_left = binom.cdf(count, T, p)
    p_right = 1 - binom.cdf(count - 1, T, p)
    return min(1.0, 2 * min(p_left, p_right))

p_normal = two_sided_binom_pvalue(count_normal, T, 1 - alpha)
p_t = two_sided_binom_pvalue(count_t, T, 1 - alpha)
p_hist = two_sided_binom_pvalue(count_hist, T, 1 - alpha)

# Backtest table
backtest_table = pd.DataFrame({
    "Method": ["Normal", "Student-t (df=5)", "Historical"],
    "Violations": [count_normal, count_t, count_hist],
    "Expected": [expected_viol, expected_viol, expected_viol],
    "cL": [cL, cL, cL],
    "cU": [cU, cU, cU],
    "p_value": [p_normal, p_t, p_hist],
    "Reject_5pct": [
        not (cL <= count_normal <= cU),
        not (cL <= count_t <= cU),
        not (cL <= count_hist <= cU)
    ]
})

print(backtest_table)

backtest_table.to_csv(f"{RESULTS_PATH}/04_portfolio_var_backtest_table.csv", index=False)

# Add yearly info
df["losses"] = df["portfolio_loss_eur"]
df["Year"] = df["Date"].dt.year

df["viol_normal"] = (df["losses"] > var_normal).astype(int)
df["viol_t"] = (df["losses"] > var_t).astype(int)
df["viol_hist"] = (df["losses"] > var_hist).astype(int)

yearly_viol = df.groupby("Year")[["viol_normal", "viol_t", "viol_hist"]].sum()
print("\nYearly violations:")
print(yearly_viol)

yearly_viol.to_csv(f"{RESULTS_PATH}/04_portfolio_yearly_violations.csv")

# Plot 1: Losses and VaR thresholds
plt.figure(figsize=(12, 5))
plt.plot(df["Date"], df["losses"], label="Portfolio Loss", linewidth=1)
plt.axhline(var_normal, linestyle="--", label="Normal VaR")
plt.axhline(var_t, linestyle="--", label="Student-t VaR")
plt.axhline(var_hist, linestyle="--", label="Historical VaR")

viol_idx = df["losses"] > var_hist
plt.scatter(df.loc[viol_idx, "Date"], df.loc[viol_idx, "losses"], s=20, label="Historical violations")

plt.title("Portfolio Losses and VaR Thresholds")
plt.xlabel("Date")
plt.ylabel("Loss (EUR)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/04_backtesting_losses_vs_var.png", dpi=300)
plt.show()

# Plot 2: Yearly violations
yearly_viol.plot(figsize=(10, 5))
plt.title("Yearly VaR Violations")
plt.xlabel("Year")
plt.ylabel("Number of violations")
plt.grid(True)
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/04_yearly_var_violations.png", dpi=300)
plt.show()

print("\nSaved files:")
print(f"{RESULTS_PATH}/04_portfolio_var_backtest_table.csv")
print(f"{RESULTS_PATH}/04_portfolio_yearly_violations.csv")
print(f"{FIGURES_PATH}/04_backtesting_losses_vs_var.png")
print(f"{FIGURES_PATH}/04_yearly_var_violations.png")