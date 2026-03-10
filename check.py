"""
check.py
--------
Prints a plain-english summary of each CSV and flags potential problems.
Run from the directory containing the CSV files.
"""

import os
import pandas as pd

CSV_FILES = [
    "TE_with_min_dg.csv",
    "cds_wobble_nucleotide_features.csv",
    "codon_aa_frequency.csv",
    "dicodon_density.csv",
    "kozak_onehot.csv",
    "nucleotide_frequency.csv",
    "kmer_features.csv",
    "lengths.csv"
]

JOIN_KEY = "Name"

# ── load all files ─────────────────────────────────────────────────────────────

dataframes = {}
for fname in CSV_FILES:
    if not os.path.isfile(fname):
        print(f"[MISSING] {fname} not found in current directory")
        continue
    dataframes[fname] = pd.read_csv(fname)

# ── per-file summary ───────────────────────────────────────────────────────────

print("=" * 60)
print("PER-FILE SUMMARY")
print("=" * 60)

for fname, df in dataframes.items():
    print(f"\n{fname}")
    print(f"  shape          : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"  columns        : {list(df.columns)}")

    # nulls
    null_counts = df.isnull().sum()
    null_cols   = null_counts[null_counts > 0]
    if null_cols.empty:
        print(f"  nulls          : none")
    else:
        print(f"  nulls          : {dict(null_cols)}")

    # duplicate Name values
    if JOIN_KEY in df.columns:
        dupes = df[JOIN_KEY].duplicated().sum()
        print(f"  duplicate Names: {dupes}")
    else:
        print(f"  [WARNING] '{JOIN_KEY}' column not found")

    # numeric summary (just min/max per column to spot obvious outliers)
    num_cols = df.select_dtypes(include="number").columns.tolist()
    if num_cols:
        desc = df[num_cols].agg(["min", "max"])
        print(f"  numeric ranges :")
        for col in num_cols:
            lo, hi = desc.loc["min", col], desc.loc["max", col]
            print(f"    {col:<35} {lo:.4g}  →  {hi:.4g}")

# ── cross-file checks ──────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("CROSS-FILE CHECKS")
print("=" * 60)

row_counts = {f: len(df) for f, df in dataframes.items()}
unique_counts = set(row_counts.values())

print("\nRow counts:")
for fname, n in row_counts.items():
    print(f"  {fname:<45} {n:,}")

if len(unique_counts) == 1:
    print("  ✓ All files have the same number of rows")
else:
    print("  ✗ Row count mismatch!")

# check Name sets match across files
print("\nName column parity (vs first file):")
files   = list(dataframes.items())
ref_f, ref_df = files[0]
ref_names = set(ref_df[JOIN_KEY])

for fname, df in files[1:]:
    if JOIN_KEY not in df.columns:
        print(f"  {fname}: missing '{JOIN_KEY}' column, skipping")
        continue
    this_names = set(df[JOIN_KEY])
    missing = ref_names - this_names
    extra   = this_names - ref_names
    if not missing and not extra:
        print(f"  ✓ {fname}")
    else:
        if missing:
            print(f"  ✗ {fname}: {len(missing)} Names in {ref_f} but not here")
        if extra:
            print(f"  ✗ {fname}: {len(extra)} Names here but not in {ref_f}")

# check for column name collisions across files
print("\nColumn name collisions across files:")
seen  = set()
found = False
for fname, df in dataframes.items():
    for col in df.columns:
        if col == JOIN_KEY:
            continue
        if col in seen:
            print(f"  ✗ '{col}' appears in multiple files (seen again in {fname})")
            found = True
        seen.add(col)
if not found:
    print("  ✓ No collisions")
