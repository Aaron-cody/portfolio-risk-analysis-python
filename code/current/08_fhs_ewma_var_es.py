import pandas as pd
import numpy as np
import os

# 08_fhs_ewma_var_es.py

BASE_PATH = "/Users/aaronroman/Desktop/QFRM"
INPUT_PATH = f"{BASE_PATH}/data/final_analysis_2026_04_09"
RESULTS_PATH = f"{BASE_PATH}/results/final_analysis_2026_04_09"

os.makedirs(RESULTS_PATH, exist_ok=True)

# =========================
# SETTINGS
# =========================
V = 1_000_000
alpha = 0.99
lam = 0.94

asset_cols = ["MSFT", "SHEL", "JPM", "^GSPC", "EURUSD=X", "LOAN"]
weights = np.array([0.20, 0.15, 0.15, 0.20, 0.10, 0.20])

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(f"{INPUT_PATH}/01_portfolio_with_returns_and_losses.csv")
returns = df[asset_cols].dropna().copy()

# =========================
# EWMA VOL FUNCTION
# =========================
def ewma_volatility(series, lam=0.94):
    series = np.asarray(series)
    sigma2 = np.zeros(len(series))
    sigma2[0] = np.var(series, ddof=1)

    for t in range(1, len(series)):
        sigma2[t] = lam * sigma2[t-1] + (1 - lam) * series[t-1]**2

    return np.sqrt(sigma2)

# =========================
# EWMA VOL FOR EACH ASSET
# =========================
ewma_vols = pd.DataFrame(index=returns.index)
std_resids = pd.DataFrame(index=returns.index)

for col in asset_cols:
    ewma_vols[col] = ewma_volatility(returns[col], lam=lam)
    std_resids[col] = returns[col] / ewma_vols[col]

# Drop initial NaNs/infs if any
std_resids = std_resids.replace([np.inf, -np.inf], np.nan).dropna()
ewma_vols = ewma_vols.loc[std_resids.index]
returns = returns.loc[std_resids.index]

# =========================
# CURRENT VOLATILITY
# =========================
current_vol = ewma_vols.iloc[-1].values

# =========================
# FILTERED HISTORICAL SIMULATION
# =========================
# Re-scale historical residuals using current vol
simulated_returns = std_resids.values * current_vol

# Portfolio simulated returns
portfolio_sim_returns = simulated_returns @ weights

# Portfolio simulated losses
portfolio_sim_losses_ret = -portfolio_sim_returns
portfolio_sim_losses_eur = V * portfolio_sim_losses_ret

# VaR / ES
var_ret = np.quantile(portfolio_sim_losses_ret, alpha)
es_ret = portfolio_sim_losses_ret[portfolio_sim_losses_ret >= var_ret].mean()

var_eur = np.quantile(portfolio_sim_losses_eur, alpha)
es_eur = portfolio_sim_losses_eur[portfolio_sim_losses_eur >= var_eur].mean()

# =========================
# RESULTS TABLE
# =========================
results = pd.DataFrame({
    "Method": ["FHS-EWMA"],
    "VaR_return": [var_ret],
    "ES_return": [es_ret],
    "VaR_EUR": [var_eur],
    "ES_EUR": [es_eur]
})

print(results)

results.to_csv(f"{RESULTS_PATH}/08_fhs_ewma_var_es_results.csv", index=False)

# Save current EWMA volatilities
current_vol_df = pd.DataFrame({
    "Asset": asset_cols,
    "Current_EWMA_Vol": current_vol
})
current_vol_df.to_csv(f"{RESULTS_PATH}/08_fhs_ewma_current_vols.csv", index=False)

print("\nSaved files:")
print(f"{RESULTS_PATH}/08_fhs_ewma_var_es_results.csv")
print(f"{RESULTS_PATH}/08_fhs_ewma_current_vols.csv")