import numpy as np
import pandas as pd

INPUT_FILE  = "TE_eIF_depletion.csv"
OUTPUT_FILE = "lengths.csv"

COLS = ["Name", "tx_length", "utr3_length", "cds_length", "utr5_length"]

df = pd.read_csv(INPUT_FILE, usecols=COLS)
df = df.dropna().reset_index(drop=True)

df["tx_length"]      = np.log1p(df["tx_length"])
df["utr5_fraction"]  = df["utr5_length"] / (df["utr5_length"] + df["cds_length"] + df["utr3_length"])
df["cds_fraction"]   = df["cds_length"]  / (df["utr5_length"] + df["cds_length"] + df["utr3_length"])
df["utr3_fraction"]  = df["utr3_length"] / (df["utr5_length"] + df["cds_length"] + df["utr3_length"])

df = df[["Name", "tx_length", "utr5_fraction", "cds_fraction", "utr3_fraction"]]

row_sums = df["utr5_fraction"] + df["cds_fraction"] + df["utr3_fraction"]
print(f"fraction row sums — min: {row_sums.min():.6f}, max: {row_sums.max():.6f}, mean: {row_sums.mean():.6f}")
if not (row_sums.between(1 - 1e-9, 1 + 1e-9)).all():
    print("WARNING: some rows do not sum to 1!")
else:
    print("all rows sum to 1 ✓")

df.to_csv(OUTPUT_FILE, index=False)
print(f"saved → {OUTPUT_FILE}  ({len(df):,} rows)")
