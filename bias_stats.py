"""
bias_stats.py
=============

Compute formal fairness metrics for the felony / non‑felony classifier:

1.  False‑positive rate (FPR) per subgroup (victim race, borough)
2.  95 % Wald confidence intervals for each FPR
3.  Two‑proportion z‑tests for the largest gaps
4.  Chi‑square test of independence across all race groups
5.  Disparate‑impact ratios (80 % rule) for felony predictions

Assumes a CSV with *at least* these columns:

    law_cat_cd       – the ground‑truth label (FELONY, MISDEMEANOR, VIOLATION)
    predicted_label  – the model’s prediction
    vic_race         – victim race (string)
    boro_nm          – borough name (string)

The script prints results and also writes:

    fpr_by_race.csv
    fpr_by_borough.csv
"""

import math
import pandas as pd
import scipy.stats as st

# -------------------------------------------------------------------
# 1.  Load predictions
# -------------------------------------------------------------------
DATA_PATH = "data/outputs/predicted_crimes_2025.csv"
df = pd.read_csv(DATA_PATH)

# -------------------------------------------------------------------
# 2.  Helper: compute FPR + CI for every group in a column
# -------------------------------------------------------------------
def fpr_table(
    data: pd.DataFrame,
    group_col: str,
    positive_label: str = "FELONY",
    min_n: int = 200,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Return a DataFrame with FPR and (1‑alpha)% Wald CI for each subgroup.

    Parameters
    ----------
    data : DataFrame
        Must contain columns `predicted_label` and `law_cat_cd`.
    group_col : str
        Column to group by (e.g. "vic_race" or "boro_nm").
    positive_label : str, default "FELONY"
        Which label is treated as the positive (adverse) event.
    min_n : int, default 200
        Skip groups with fewer than this many (FP+TN) instances to avoid
        absurdly wide confidence intervals.
    alpha : float, default 0.05
        Significance level for CIs (0.05 → 95 % CI).
    """
    rows = []
    z = st.norm.ppf(1 - alpha / 2)  # two‑sided z‑score
    for group, g in data.groupby(group_col):
        # False positive = predicted FELONY but truth is not FELONY
        fp = ((g["predicted_label"] == positive_label) &
              (g["law_cat_cd"] != positive_label)).sum()
        # True negative = predicted not FELONY and truth not FELONY
        tn = ((g["predicted_label"] != positive_label) &
              (g["law_cat_cd"] != positive_label)).sum()
        n = fp + tn
        if n < min_n:
            continue
        fpr = fp / n
        # Wald standard error
        se = math.sqrt(fpr * (1 - fpr) / n)
        ci_low = max(0, fpr - z * se)
        ci_high = min(1, fpr + z * se)
        rows.append({"group": group, "n": n, "fp": fp, "tn": tn,
                     "fpr": fpr, "ci_low": ci_low, "ci_high": ci_high})
    return pd.DataFrame(rows).sort_values("fpr").reset_index(drop=True)

# -------------------------------------------------------------------
# 3.  Compute FPR tables and save
# -------------------------------------------------------------------
race_fpr = fpr_table(df, "vic_race")
borough_fpr = fpr_table(df, "boro_nm")

race_fpr.to_csv("fpr_by_race.csv", index=False)
borough_fpr.to_csv("fpr_by_borough.csv", index=False)

print("False‑positive rate by victim race (95 % CI)")
print(race_fpr[["group", "fpr", "ci_low", "ci_high", "n"]])
print("\nFalse‑positive rate by borough (95 % CI)")
print(borough_fpr[["group", "fpr", "ci_low", "ci_high", "n"]])

# -------------------------------------------------------------------
# 4.  Two‑proportion z‑tests for biggest gaps
# -------------------------------------------------------------------
def two_prop_z(fp1, n1, fp2, n2):
    """
    Compute z‑score and two‑sided p‑value for difference in proportions.
    """
    p1 = fp1 / n1
    p2 = fp2 / n2
    p_pool = (fp1 + fp2) / (n1 + n2)
    se = math.sqrt(p_pool * (1 - p_pool) * (1 / n1 + 1 / n2))
    z = (p1 - p2) / se
    p = 2 * (1 - st.norm.cdf(abs(z)))
    return p1, p2, z, p

# Identify “best” and “worst” race groups
best_race = race_fpr.iloc[0]
worst_race = race_fpr.iloc[-1]

p1, p2, z, p = two_prop_z(best_race.fp, best_race.n,
                          worst_race.fp, worst_race.n)
print(f"\nVictim race gap: {best_race.group} vs {worst_race.group}")
print(f"  FPR difference: {p2 - p1:.3f}  z = {z:.1f}  p = {p:.3g}")

# Same for borough
best_boro = borough_fpr.iloc[0]
worst_boro = borough_fpr.iloc[-1]

p1, p2, z, p = two_prop_z(best_boro.fp, best_boro.n,
                          worst_boro.fp, worst_boro.n)
print(f"\nBorough gap: {best_boro.group} vs {worst_boro.group}")
print(f"  FPR difference: {p2 - p1:.3f}  z = {z:.1f}  p = {p:.3g}")

# -------------------------------------------------------------------
# 5.  Chi‑square test of independence (race × (FP,TN))
# -------------------------------------------------------------------
contingency = pd.DataFrame({
    "FP": race_fpr.set_index("group")["fp"],
    "TN": race_fpr.set_index("group")["tn"]
})
chi2, p_chi, dof, _ = st.chi2_contingency(contingency.values)
print(f"\nChi‑square test (race × error type): "
      f"χ² = {chi2:.1f}, df = {dof}, p = {p_chi:.3g}")

# -------------------------------------------------------------------
# 6.  Disparate‑impact ratios (80 % rule)
# -------------------------------------------------------------------
def disparate_impact(rate_series):
    """
    Return min / max ratio (impact value).
    Lower values < 0.8 may indicate disparate impact.
    """
    return rate_series.min() / rate_series.max()

pred_rate_race = (
    df.groupby("vic_race")
      .apply(lambda g: (g["predicted_label"] == "FELONY").mean())
      .dropna()
)
impact_race = disparate_impact(pred_rate_race)

pred_rate_boro = (
    df.groupby("boro_nm")
      .apply(lambda g: (g["predicted_label"] == "FELONY").mean())
      .dropna()
)
impact_boro = disparate_impact(pred_rate_boro)

print(f"\nDisparate‑impact (victim race): {impact_race:.2f}")
print(f"Disparate‑impact (borough):     {impact_boro:.2f}")

print("\nDone.  Detailed CSVs saved to disk.")
