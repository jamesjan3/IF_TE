import pandas as pd

# ── Loading data ──────────────────────────────────────────────────────────────

df = pd.read_csv("TE_eIF_depletion.csv")

print(df.head())

# ── Nucleotide frequencies ────────────────────────────────────────────────────

# Function: get_regions
# - Slices the sequence into regions
# - Adds them as new columns
def get_regions(row):
    seq = row["tx_sequence"]

    utr5_len = int(row["utr5_length"])
    cds_len  = int(row["cds_length"])
    utr3_len = int(row["utr3_length"])

    utr5 = seq[:utr5_len]
    cds  = seq[utr5_len:utr5_len + cds_len]
    utr3 = seq[utr5_len + cds_len:utr5_len + cds_len + utr3_len]

    return pd.Series([utr5, cds, utr3])


# Creates copy of df
df_features = df.copy()

# Drops any row with NaN in any column
df_features = df_features.dropna().reset_index(drop=True)

# Applies get_regions function
df_features[["utr5_seq", "cds_seq", "utr3_seq"]] = df_features.apply(get_regions, axis=1)

print(df_features.head())

# Function: nucleotide_percentages
# - calculates normalised nucleotide frequency
def nucleotide_percentages(sequence):
    sequence = sequence.upper()
    length = len(sequence)

    if length == 0:
        return pd.Series([0, 0, 0, 0])

    A = sequence.count("A") / length
    C = sequence.count("C") / length
    G = sequence.count("G") / length
    T = sequence.count("T") / length

    return pd.Series([A, C, G, T])


# Computes nucleotide frequencies for each region:

# 5′ UTR
df_features[["utr5_A_pct", "utr5_C_pct", "utr5_G_pct", "utr5_T_pct"]] = \
    df_features["utr5_seq"].apply(nucleotide_percentages)

# CDS
df_features[["cds_A_pct", "cds_C_pct", "cds_G_pct", "cds_T_pct"]] = \
    df_features["cds_seq"].apply(nucleotide_percentages)

# 3′ UTR
df_features[["utr3_A_pct", "utr3_C_pct", "utr3_G_pct", "utr3_T_pct"]] = \
    df_features["utr3_seq"].apply(nucleotide_percentages)

# Full transcript
df_features[["tx_A_pct", "tx_C_pct", "tx_G_pct", "tx_T_pct"]] = \
    df_features["tx_sequence"].apply(nucleotide_percentages)

# Checking output
print(df_features.head())

# ── Saving results as .csv ────────────────────────────────────────────────────

# 5′ UTR
df_features["utr5_total_pct"] = df_features[["utr5_A_pct","utr5_C_pct","utr5_G_pct","utr5_T_pct"]].sum(axis=1)

# CDS
df_features["cds_total_pct"] = df_features[["cds_A_pct","cds_C_pct","cds_G_pct","cds_T_pct"]].sum(axis=1)

# 3′ UTR
df_features["utr3_total_pct"] = df_features[["utr3_A_pct","utr3_C_pct","utr3_G_pct","utr3_T_pct"]].sum(axis=1)

# Full transcript
df_features["tx_total_pct"] = df_features[["tx_A_pct","tx_C_pct","tx_G_pct","tx_T_pct"]].sum(axis=1)

# Extracts nucleotide frequency results for 5', CDS and 3' and identifier
feature_columns = [
    "utr5_A_pct","utr5_C_pct","utr5_G_pct","utr5_T_pct",
    "cds_A_pct","cds_C_pct","cds_G_pct","cds_T_pct",
    "utr3_A_pct","utr3_C_pct","utr3_G_pct","utr3_T_pct",
    "tx_A_pct","tx_C_pct","tx_G_pct","tx_T_pct"
]

df_nuc_features = df_features[["Name"] + feature_columns]

# Save to CSV
df_nuc_features.to_csv("nucleotide_features.csv", index=False)

print("CSV saved")

# ── Sanity check: Do nucleotide frequencies sum up to 1 per region? ───────────

# Sets a small tolerance for floating point rounding errors
tolerance = 0.0001

# Regions to check
regions = ["utr5", "cds", "utr3", "tx"]

# Iterates and prints DataFrame for each region where sum != 1
for region in regions:
    total_col = f"{region}_total_pct"

    # Filters rows where sum is outside 1 ± tolerance
    invalid_rows = df_features[
        (df_features[total_col] < 1 - tolerance) |
        (df_features[total_col] > 1 + tolerance)
    ][["Name", total_col]]

    if not invalid_rows.empty:
        print(f"\nRegion: {region.upper()} — rows not summing to 1:")
        print(invalid_rows)

# ── Wobble position: nucleotide percentage ────────────────────────────────────

# Function: get_wobble_positions
# - Returns a string of the 3rd nucleotide of each codon in the CDS.
def get_wobble_positions(cds_seq):
    cds_seq = cds_seq.upper() if pd.notna(cds_seq) else ""
    # Taking every 3rd nucleotide starting at index 2 (0-based indexing)
    wobble_seq = cds_seq[2::3]
    return wobble_seq


# Extracts wobble positions
df_features["cds_wobble_seq"] = df_features["cds_seq"].apply(get_wobble_positions)

# Calculates nucleotide frequencies
df_features[["cds_wobble_A_pct","cds_wobble_C_pct","cds_wobble_G_pct","cds_wobble_T_pct"]] = \
    df_features["cds_wobble_seq"].apply(nucleotide_percentages)

# ── Saving results as .csv ────────────────────────────────────────────────────

wobble_columns = ["cds_wobble_A_pct","cds_wobble_C_pct","cds_wobble_G_pct","cds_wobble_T_pct"]
df_wobble_features = df_features[["Name"] + wobble_columns]

df_wobble_features.to_csv("cds_wobble_nucleotide_features.csv", index=False)

print("CSV saved")

# ── Sanity check: Is any CDS not divisible by 3? ─────────────────────────────

# Creates boolean column: True if CDS length is divisible by 3
df_features["cds_div3"] = df_features["cds_seq"].str.len() % 3 == 0

# Counts how many are not divisible by 3
num_not_div3 = (~df_features["cds_div3"]).sum()

print(f"Number of CDS sequences NOT divisible by 3: {num_not_div3}")

# ── Sanity check: Do nucleotide frequencies in wobble position sum up to 1? ───

# First, compute total percentage at CDS wobble positions
df_features["cds_wobble_total_pct"] = df_features[[
    "cds_wobble_A_pct","cds_wobble_C_pct","cds_wobble_G_pct","cds_wobble_T_pct"
]].sum(axis=1)

# Set a small tolerance for floating point rounding errors
tolerance = 0.01

# Count how many rows do not sum to 100%
num_invalid_wobble = ((df_features["cds_wobble_total_pct"] < 1 - tolerance) |
                      (df_features["cds_wobble_total_pct"] > 1 + tolerance)).sum()

total_rows = len(df_features)
percent_invalid_wobble = (num_invalid_wobble / total_rows) * 100

print(f"Number of CDS wobble rows not summing to 1: {num_invalid_wobble} ({percent_invalid_wobble:.2f}%)")

# Optional: see the rows explicitly
df_invalid_wobble = df_features.loc[
    (df_features["cds_wobble_total_pct"] < 100 - tolerance) |
    (df_features["cds_wobble_total_pct"] > 100 + tolerance),
    ["Name","cds_wobble_total_pct"]
]

print(df_invalid_wobble)
