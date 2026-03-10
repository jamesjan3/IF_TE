"""
merge.py
--------
Merges all CSVs on the 'Name' column and writes merged_rna_features.csv.
Drops fully duplicate rows before writing.
Run from the directory containing the CSV files.
"""

import pandas as pd

CSV_FILES = [
    "TE_with_min_dg.csv",
    "cds_wobble_nucleotide_features.csv",
    "codon_aa_frequency.csv",
    "dicodon_density.csv",
    "kozak_onehot.csv",
    "nucleotide_frequency.csv",
    "kmer_features.csv",
    "lengths.csv",
]

JOIN_KEY    = "Name"
OUTPUT_FILE = "merged_features.csv"

merged = None

for fname in CSV_FILES:
    df = pd.read_csv(fname)
    if merged is None:
        merged = df
    else:
        merged = merged.merge(df, on=JOIN_KEY, how="outer")
    print(f"merged {fname:45}  shape so far: {merged.shape}")

# drop fully duplicate rows
before = len(merged)
merged = merged.drop_duplicates()
dropped = before - len(merged)
if dropped:
    print(f"\ndropped {dropped} fully duplicate rows")

merged.to_csv(OUTPUT_FILE, index=False)
print(f"\nsaved → {OUTPUT_FILE}  ({merged.shape[0]:,} rows x {merged.shape[1]} columns)")
