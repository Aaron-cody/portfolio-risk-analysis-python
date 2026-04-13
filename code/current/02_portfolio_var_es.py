import pandas as pd
import numpy as np
import os
from scipy.stats import norm, t

# 02_portfolio_var_es.py

BASE_PATH = "/Users/aaronroman/Desktop/QFRM"
INPUT_PATH = f"{BASE_PATH}/data/final_analysis_2026_04_09"
OUTPUT_PATH = f"{BASE_PATH}/results/final_analysis_2026_04_09"

os.makedirs(OUTPUT_PATH, exist_ok=True)



# Load rebuilt portfolio dataset
df = pd.read_csv(f"{INPUT_PATH}/01_portfolio_with_returns_and_losses.csv")

# Portfolio value
V = 1_000_000

# Confidence level
alpha = 0.99

# Portfolio return series
r = df["portfolio_return"].dropna()

# =========================
# 1. NORMAL VaR and ES
# =========================
mu = r.mean()
sigma = r.std(ddof=1)

z_alpha = norm.ppf(alpha)

var_normal_ret = -mu + z_alpha * sigma
es_normal_ret = -mu + sigma * (norm.pdf(z_alpha) / (1 - alpha))

var_normal_eur = V * var_normal_ret
es_normal_eur = V * es_normal_ret

# =========================
# 2. STUDENT-t VaR and ES
# =========================
nu = 5

t_alpha = t.ppf(alpha, df=nu)
scale_adj = np.sqrt((nu - 2) / nu)

var_t_ret = -mu + sigma * scale_adj * t_alpha

pdf_t = t.pdf(t_alpha, df=nu)
es_t_ret = -mu + sigma * scale_adj * (
    (pdf_t / (1 - alpha)) * ((nu + t_alpha**2) / (nu - 1))
)

var_t_eur = V * var_t_ret
es_t_eur = V * es_t_ret

# =========================
# 3. HISTORICAL SIMULATION
# =========================
losses_ret = -r
losses_eur = V * losses_ret

var_hist_ret = np.quantile(losses_ret, alpha)
es_hist_ret = losses_ret[losses_ret >= var_hist_ret].mean()

var_hist_eur = np.quantile(losses_eur, alpha)
es_hist_eur = losses_eur[losses_eur >= var_hist_eur].mean()

# =========================
# RESULTS TABLE
# =========================
results = pd.DataFrame({
    "Method": ["Normal", "Student-t (df=5)", "Historical"],
    "VaR_return": [var_normal_ret, var_t_ret, var_hist_ret],
    "ES_return": [es_normal_ret, es_t_ret, es_hist_ret],
    "VaR_EUR": [var_normal_eur, var_t_eur, var_hist_eur],
    "ES_EUR": [es_normal_eur, es_t_eur, es_hist_eur]
})

print(results)

# Save results
results.to_csv(f"{OUTPUT_PATH}/02_portfolio_var_es_results.csv", index=False)

print("\nSaved file:")
print(f"{OUTPUT_PATH}/02_portfolio_var_es_results.csv")