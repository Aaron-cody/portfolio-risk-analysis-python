import pandas as pd
import os
import matplotlib.pyplot as plt

# 09_final_method_comparison.py

BASE_PATH = "/Users/aaronroman/Desktop/QFRM"
RESULTS_PATH = f"{BASE_PATH}/results/final_analysis_2026_04_09"
FIGURES_PATH = f"{BASE_PATH}/figures/final_analysis_2026_04_09"

os.makedirs(RESULTS_PATH, exist_ok=True)
os.makedirs(FIGURES_PATH, exist_ok=True)

# =========================
# LOAD RESULT FILES
# =========================
df_static = pd.read_csv(f"{RESULTS_PATH}/02_portfolio_var_es_results.csv")
df_garch = pd.read_csv(f"{RESULTS_PATH}/07_garch_ccc_var_es_results.csv")
df_fhs = pd.read_csv(f"{RESULTS_PATH}/08_fhs_ewma_var_es_results.csv")

# Combine all methods
comparison = pd.concat([df_static, df_garch, df_fhs], ignore_index=True)

# Clean ordering
method_order = [
    "Normal",
    "Student-t (df=4)",
    "Student-t (df=5)",
    "Historical",
    "GARCH-CCC Normal",
    "GARCH-CCC Student-t (df=4)",
    "FHS-EWMA"
]

comparison["order"] = comparison["Method"].apply(
    lambda x: method_order.index(x) if x in method_order else 999
)
comparison = comparison.sort_values("order").drop(columns="order")

print(comparison)

# Save final comparison table
comparison.to_csv(f"{RESULTS_PATH}/09_final_method_comparison.csv", index=False)

# =========================
# PLOT 1: VaR comparison
# =========================
plt.figure(figsize=(10, 5))
plt.bar(comparison["Method"], comparison["VaR_EUR"])
plt.title("Comparison of 1-day VaR Across Methods")
plt.xlabel("Method")
plt.ylabel("VaR (EUR)")
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/09_var_comparison.png", dpi=300)
plt.show()

# =========================
# PLOT 2: ES comparison
# =========================
plt.figure(figsize=(10, 5))
plt.bar(comparison["Method"], comparison["ES_EUR"])
plt.title("Comparison of 1-day ES Across Methods")
plt.xlabel("Method")
plt.ylabel("ES (EUR)")
plt.xticks(rotation=45, ha="right")
plt.grid(True, axis="y")
plt.tight_layout()
plt.savefig(f"{FIGURES_PATH}/09_es_comparison.png", dpi=300)
plt.show()

print("\nSaved files:")
print(f"{RESULTS_PATH}/09_final_method_comparison.csv")
print(f"{FIGURES_PATH}/09_var_comparison.png")
print(f"{FIGURES_PATH}/09_es_comparison.png")