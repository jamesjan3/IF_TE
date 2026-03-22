import numpy as np
import pandas as pd
from seqfold import dg


#set temperature and convert to RNA uppercase
TEMP_C = 37.0

def dg_window_or_nan(seq):
    # seq is already cleaned to RNA uppercase
    if not seq or any(b not in "ACGU" for b in seq):
        return np.nan
    return float(dg(seq, temp=TEMP_C))

#Define the function for the 60nt centered on the start codon

def start_centered_window(seq, start_idx, win=60):
    if not seq or pd.isna(start_idx):
        return ""
    start_idx = int(start_idx)
    w0 = max(0, start_idx - win//2)
    w1 = min(len(seq), w0 + win)
    w0 = max(0, w1 - win)
    return seq[w0:w1]


#Load CSV file and drop any NaN values
df = pd.read_csv("TE_eIF_depletion.csv").dropna().reset_index(drop=True)


#Replace T with U for switching from DNA to RNA bases
df["tx_sequence"] = (
    df["tx_sequence"].astype(str).str.upper().str.replace("T", "U", regex=False)
)

#First 60 nt min Delta G

w5 = df["tx_sequence"].str.slice(0, 60)

# valid if only A/C/G/U and length > 0 (handles shorter transcripts too)
valid = w5.str.fullmatch(r"[ACGU]+")

out = np.full(len(df), np.nan, dtype=float)

valid_w = w5[valid].tolist()          # plain Python list (faster iteration than Series)
out_valid = [float(dg(s, temp=TEMP_C)) for s in valid_w]

out[valid.to_numpy()] = out_valid
df["min_dg_5p60"] = out

#Min Delta G starting centered on the Start codon

def start_centered_window(seq, utr5_len, win=60):
    seq = str(seq).upper().replace("T", "U")

    if not seq or pd.isna(utr5_len):
        return ""

    start_idx = int(utr5_len)      # A of AUG (0-based)
    half = win // 2                # 30 for 60-nt window

    w0 = max(0, start_idx - half)
    w1 = min(len(seq), w0 + win)

    # shift left if truncated at end
    w0 = max(0, w1 - win)

    return seq[w0:w1]


w_start = df.apply(
    lambda r: start_centered_window(r["tx_sequence"], r["utr5_length"], win=60),
    axis=1
)

valid_start = w_start.str.fullmatch(r"[ACGU]+")

out_start = np.full(len(df), np.nan, dtype=float)

valid_w = w_start[valid_start].tolist()
out_valid = [float(dg(s, temp=TEMP_C)) for s in valid_w]

out_start[valid_start.to_numpy()] = out_valid

df["min_dg_startCentered60"] = out_start


#Save the necessary columns to a data frame

cols_to_save = [
    "Name",
    "eIF3d_control_logTE",
    "eIF3d_depletion_logTE",
    "eIF4e_control_logTE",
    "eIF4e_depletion_logTE",
    "min_dg_5p60",
    "min_dg_startCentered60"
]

df_out = df[cols_to_save]

df_out.to_csv("TE_with_min_dg.csv", index=False)

print("Saved: TE_with_min_dg.csv")