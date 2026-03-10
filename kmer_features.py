"""
Memory-safe k-mer feature extraction (streaming version)
Includes end-of-run sanity check.

- Streams rows directly to CSV
- Precomputes all possible k-mers once
- Keeps memory usage small
"""

import itertools
import pandas as pd
import csv
from collections import Counter

# ─────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────
CSV_PATH = "TE_eIF_depletion.csv"
OUT_PATH = "kmer_features.csv"
regions  = ["utr5", "cds", "utr3"]
k_range  = range(2, 5)
NUCLEOTIDES = "ACGT"

# ─────────────────────────────────────────────────────────────
# Load data (drop NaNs early)
# ─────────────────────────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df.dropna(subset=["utr5_length", "cds_length", "utr3_length", "tx_sequence"])
df = df.reset_index(drop=True)

n_total = len(df)
print(f"Loaded {n_total} transcripts.", flush=True)

# ─────────────────────────────────────────────────────────────
# Precompute all possible k-mers ONCE
# ─────────────────────────────────────────────────────────────
ALL_KMERS = {
    k: ["".join(p) for p in itertools.product(NUCLEOTIDES, repeat=k)]
    for k in k_range
}

total_features = sum(len(ALL_KMERS[k]) for k in k_range) * len(regions)
print(f"Total features per transcript: {total_features}", flush=True)

# ─────────────────────────────────────────────────────────────
# Region extraction
# ─────────────────────────────────────────────────────────────
def extract_regions(row):
    seq = str(row["tx_sequence"])
    u5  = int(row["utr5_length"])
    cds = int(row["cds_length"])

    return {
        "utr5": seq[:u5],
        "cds":  seq[u5 : u5 + cds],
        "utr3": seq[u5 + cds:],
    }

# ─────────────────────────────────────────────────────────────
# Main streaming loop
# ─────────────────────────────────────────────────────────────
print("Starting k-mer computation...\n", flush=True)

with open(OUT_PATH, "w", newline="") as f:
    writer = None

    for i, (_, row) in enumerate(df.iterrows()):
        seqs = extract_regions(row)
        feat_row = {"Name": row["Name"]}

        for region in regions:
            sequence = seqs[region]

            for k in k_range:
                kmers_list = ALL_KMERS[k]

                counts = Counter(
                    sequence[j:j+k]
                    for j in range(len(sequence) - k + 1)
                )

                total = sum(counts.values())

                for kmer in kmers_list:
                    freq = counts.get(kmer, 0) / total if total > 0 else 0.0
                    feat_row[f"{region}_k{k}_{kmer}"] = freq

        # Initialize writer once with header
        if writer is None:
            writer = csv.DictWriter(f, fieldnames=feat_row.keys())
            writer.writeheader()

        writer.writerow(feat_row)

        if (i + 1) % 200 == 0 or (i + 1) == n_total:
            print(f"[{i+1:>6} / {n_total}] transcripts processed", flush=True)

print("\nFinished writing CSV.\n")

# ─────────────────────────────────────────────────────────────
# Sanity Check
# ─────────────────────────────────────────────────────────────
print("Sanity check (frequency sums for first transcript):")

df_check = pd.read_csv(OUT_PATH, nrows=1)

for region in regions:
    for k in k_range:
        cols = [c for c in df_check.columns if c.startswith(f"{region}_k{k}_")]
        row_sum = df_check[cols].iloc[0].sum()
        status = "✓" if abs(row_sum - 1.0) < 1e-6 else "✗"
        print(f"  {status}  {region} k={k} freq sum: {row_sum:.6f}")

print("\nDone.")
print(f"Saved to: {OUT_PATH}")
