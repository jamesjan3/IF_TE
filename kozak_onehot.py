"""
Streaming one-hot encoding for Kozak positions:
-3, -2, -1, +4, +5

Drops T as reference category.
Outputs 3x5=15 predictors per transcript.
"""

import pandas as pd
import csv

CSV_PATH = "TE_eIF_depletion.csv"
OUT_PATH = "kozak_onehot.csv"

positions = {
    "-3": -3,
    "-2": -2,
    "-1": -1,
    "+4":  3,
    "+5":  4,
}

NUCLEOTIDES = ["A", "C", "G"]  # T is reference

# ─────────────────────────────────────────
# load
# ─────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["utr5_length", "tx_sequence"])
df = df.reset_index(drop=True)

print(f"Loaded {len(df)} transcripts.", flush=True)

# ─────────────────────────────────────────
# streaming write
# ─────────────────────────────────────────
with open(OUT_PATH, "w", newline="") as f:
    writer = None

    for i, (_, row) in enumerate(df.iterrows()):
        seq = str(row["tx_sequence"])
        u5  = int(row["utr5_length"])

        feat_row = {"Name": row["Name"]}

        valid = True

        for label, offset in positions.items():
            idx = u5 + offset

            if idx < 0 or idx >= len(seq):
                valid = False
                base = None
            else:
                base = seq[idx]

            for nuc in NUCLEOTIDES:
                colname = f"{label}_{nuc}"
                feat_row[colname] = 1 if base == nuc else 0

        # Initialize header once
        if writer is None:
            writer = csv.DictWriter(f, fieldnames=feat_row.keys())
            writer.writeheader()

        writer.writerow(feat_row)

        if (i + 1) % 1000 == 0:
            print(f"[{i+1}] processed", flush=True)

print("Done.")
print(f"Saved to {OUT_PATH}")
