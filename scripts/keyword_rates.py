import re
from pathlib import Path
import pandas as pd
STEER_DIR = Path("results/steering")
PATTERN = re.compile(r"\b(?:pain|painful|hurt|hurts|hurting)\b", re.IGNORECASE)
frames = []
for f in sorted(STEER_DIR.glob("4.2_steering_*_L*.csv")):
    df = pd.read_csv(f)
    df["file"] = f.name
    frames.append(df)
if not frames:
    raise SystemExit(f"No steering CSVs found in {STEER_DIR.resolve()}")
A = pd.concat(frames, ignore_index=True)

A["hit"] = A["generation"].fillna("").astype(str).apply(lambda t: bool(PATTERN.search(t)))

by_coeff = (A.groupby(["file", "coeff"])["hit"]
              .agg(rate="mean", hits="sum", n="size")
              .reset_index())
by_coeff["rate"] = (by_coeff["rate"] * 100).round(1)
print("Percent of generations containing a pain/hurt word:\n")
print(by_coeff.to_string(index=False))

pos = A[A["coeff"] > 0]
pooled = (pos.groupby("file")["hit"]
             .agg(rate="mean", hits="sum", n="size")
             .reset_index())
pooled["rate"] = (pooled["rate"] * 100).round(1)
print("\nPositive coefficients pooled:\n")
print(pooled.to_string(index=False))

by_coeff.to_csv(STEER_DIR / "keyword_rates_by_coeff.csv", index=False)
pooled.to_csv(STEER_DIR / "keyword_rates_pooled.csv", index=False)
print("\nSaved keyword_rates_by_coeff.csv and keyword_rates_pooled.csv")