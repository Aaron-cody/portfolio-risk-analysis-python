import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import t, probplot

# 03_student_t_qqplot.py

BASE_PATH = "/Users/aaronroman/Desktop/QFRM"
INPUT_PATH = f"{BASE_PATH}/data/final_analysis_2026_04_09"
OUTPUT_PATH = f"{BASE_PATH}/figures/final_analysis_2026_04_09"

os.makedirs(OUTPUT_PATH, exist_ok=True)

# Load rebuilt portfolio dataset
df = pd.read_csv(f"{INPUT_PATH}/01_portfolio_with_returns_and_losses.csv")

# Portfolio return series
r = df["portfolio_return"].dropna()

# Standardize returns
mu = r.mean()
sigma = r.std(ddof=1)
z = (r - mu) / sigma

dfs = [3, 4, 5, 6]

for nu in dfs:
    plt.figure(figsize=(6, 4))
    probplot(z, dist=t, sparams=(nu,), plot=plt)
    plt.title(f"QQ-plot vs Student-t (df={nu})")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/03_qqplot_student_t_df_{nu}.png", dpi=300)
    plt.show()

print("Saved QQ-plots to:")
print(OUTPUT_PATH)