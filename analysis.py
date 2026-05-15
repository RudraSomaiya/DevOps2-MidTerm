import pandas as pd
 
# ── Load Dataset ────────────────────────────────────────────────────
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
 
print("=" * 55)
print("       TITANIC DATASET — ANALYSIS REPORT")
print("=" * 55)
 
# ── 1. Total Records ────────────────────────────────────────────────
total = len(df)
print(f"\n[1] TOTAL RECORDS")
print(f"    Total Passengers : {total}")
print(f"    Total Columns    : {df.shape[1]}")
 
# ── 2. Survival Rate ─────────────────────────────────────────────────
survived     = df["Survived"].sum()
not_survived = total - survived
survival_pct = (survived / total) * 100
print(f"\n[2] SURVIVAL RATE")
print(f"    Survived         : {survived}")
print(f"    Did Not Survive  : {not_survived}")
print(f"    Survival Rate    : {survival_pct:.2f}%")
 
# ── 3. Average Age ───────────────────────────────────────────────────
avg_age = df["Age"].mean()
missing_age = df["Age"].isnull().sum()
print(f"\n[3] AVERAGE AGE")
print(f"    Average Age      : {avg_age:.2f} years")
print(f"    Missing Age vals : {missing_age} (excluded from avg)")
 
# ── 4. Class-wise Distribution ────────────────────────────────────────
class_counts = df["Pclass"].value_counts().sort_index()
class_surv   = df.groupby("Pclass")["Survived"].agg(["sum","count","mean"])
class_surv.columns = ["Survived","Total","Rate"]
class_surv["Rate"] = (class_surv["Rate"] * 100).round(2)
print(f"\n[4] CLASS-WISE DISTRIBUTION")
print(f"    Class 1 (First)  : {class_counts[1]} passengers")
print(f"    Class 2 (Second) : {class_counts[2]} passengers")
print(f"    Class 3 (Third)  : {class_counts[3]} passengers")
print(f"\n[5] CLASS-WISE SURVIVAL RATE")
for cls in [1, 2, 3]:
    row = class_surv.loc[cls]
    print(f"    Class {cls}: {int(row.Survived)}/{int(row.Total)} survived  ({row.Rate:.2f}%)")
 
print("\n" + "=" * 55)
