import pandas as pd
import numpy as np
import os
from scipy.stats import norm, t
from arch import arch_model

# 07_garch_ccc_var_es.py

BASE_PATH = "/Users/aaronroman/Desktop/QFRM"
INPUT_PATH = f"{BASE_PATH}/data/final_analysis_2026_04_09"
RESULTS_PATH = f"{BASE_PATH}/results/final_analysis_2026_04_09"

os.makedirs(RESULTS_PATH, exist_ok=True)

# =========================
# SETTINGS
# =========================
V = 1_000_000
alpha = 0.99
nu = 4   # use your chosen Student-t df if needed later

asset_cols = ["MSFT", "SHEL", "JPM", "^GSPC", "EURUSD=X", "LOAN"]

weights = np.array([0.20, 0.15, 0.15, 0.20, 0.10, 0.20])

# =========================
# LOAD DATA
# =========================
df = pd.read_csv(f"{INPUT_PATH}/01_portfolio_with_returns_and_losses.csv")

returns = df[asset_cols].dropna().copy()

# =========================
# FIT GARCH(1,1) PER ASSET
# =========================
cond_vols = pd.DataFrame(index=returns.index)
std_resids = pd.DataFrame(index=returns.index)
garch_summary = []

for col in asset_cols:
    # arch package works better when returns are scaled
    r = returns[col] * 100

    model = arch_model(
        r,
        mean="Constant",
        vol="GARCH",
        p=1,
        q=1,
        dist="normal"
    )

    res = model.fit(disp="off")

    # Conditional volatility, convert back to decimal scale
    cond_vols[col] = res.conditional_volatility / 100.0

    # Standardized residuals
    std_resids[col] = res.std_resid

    params = res.params
    garch_summary.append({
        "Asset": col,
        "mu": params.get("mu", np.nan),
        "omega": params.get("omega", np.nan),
        "alpha[1]": params.get("alpha[1]", np.nan),
        "beta[1]": params.get("beta[1]", np.nan)
    })

garch_summary_df = pd.DataFrame(garch_summary)
garch_summary_df.to_csv(f"{RESULTS_PATH}/07_garch_ccc_parameters.csv", index=False)

# =========================
# CCC CORRELATION MATRIX
# =========================
# Constant correlation from standardized residuals
P = std_resids.corr()

# Last conditional volatilities
last_vols = cond_vols.iloc[-1].values

# Diagonal volatility matrix
Delta = np.diag(last_vols)

# Conditional covariance matrix
Sigma_t = Delta @ P.values @ Delta

# =========================
# CONDITIONAL PORTFOLIO MEAN/VOL
# =========================
mu_vec = returns.mean().values
mu_p = weights @ mu_vec
sigma_p = np.sqrt(weights @ Sigma_t @ weights)

# =========================
# GARCH-CCC NORMAL VaR / ES
# =========================
z_alpha = norm.ppf(alpha)

var_ret_normal = -mu_p + z_alpha * sigma_p
es_ret_normal = -mu_p + sigma_p * (norm.pdf(z_alpha) / (1 - alpha))

var_eur_normal = V * var_ret_normal
es_eur_normal = V * es_ret_normal

# =========================
# OPTIONAL: GARCH-CCC STUDENT-t VaR / ES
# =========================
t_alpha = t.ppf(alpha, df=nu)
scale_adj = np.sqrt((nu - 2) / nu)

var_ret_t = -mu_p + sigma_p * scale_adj * t_alpha

pdf_t = t.pdf(t_alpha, df=nu)
es_ret_t = -mu_p + sigma_p * scale_adj * (
    (pdf_t / (1 - alpha)) * ((nu + t_alpha**2) / (nu - 1))
)

var_eur_t = V * var_ret_t
es_eur_t = V * es_ret_t

# =========================
# RESULTS TABLE
# =========================
results = pd.DataFrame({
    "Method": ["GARCH-CCC Normal", f"GARCH-CCC Student-t (df={nu})"],
    "VaR_return": [var_ret_normal, var_ret_t],
    "ES_return": [es_ret_normal, es_ret_t],
    "VaR_EUR": [var_eur_normal, var_eur_t],
    "ES_EUR": [es_eur_normal, es_eur_t]
})

print(results)

results.to_csv(f"{RESULTS_PATH}/07_garch_ccc_var_es_results.csv", index=False)

# Save CCC correlation matrix
P.to_csv(f"{RESULTS_PATH}/07_garch_ccc_correlation_matrix.csv")

# Save conditional covariance matrix
sigma_df = pd.DataFrame(Sigma_t, index=asset_cols, columns=asset_cols)
sigma_df.to_csv(f"{RESULTS_PATH}/07_garch_ccc_covariance_matrix.csv")

print("\nSaved files:")
print(f"{RESULTS_PATH}/07_garch_ccc_var_es_results.csv")
print(f"{RESULTS_PATH}/07_garch_ccc_parameters.csv")
print(f"{RESULTS_PATH}/07_garch_ccc_correlation_matrix.csv")
print(f"{RESULTS_PATH}/07_garch_ccc_covariance_matrix.csv")