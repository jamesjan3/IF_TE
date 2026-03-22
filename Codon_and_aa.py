
import pandas as pd
import numpy as np
from collections import Counter

df = pd.read_csv("TE_eIF_depletion.csv")


#Find Rows with NaN values and drop them
problem_rows = df[
    df["utr5_length"].isna()
    | df["cds_length"].isna()
    | df["tx_sequence"].isna()
]

print(f"Problematic rows: {len(problem_rows)}")
print(problem_rows[["Name", "utr5_length", "cds_length"]].head())

df = df.dropna().reset_index(drop=True)

#Extract codons without start codons

def extract_cds(row):
    start = int(row["utr5_length"]) + 3
    end = start + int(row["cds_length"])
    return row["tx_sequence"][start:end]

df["cds_sequence"] = df.apply(extract_cds, axis=1)

#Define the bases and a function for codon frequencies

bases = ["A", "C", "G", "T"]
all_codons = [a + b + c for a in bases for b in bases for c in bases]


def codon_frequencies(cds):
    codons = [cds[i:i+3] for i in range(0, len(cds) - 2, 3)]
    counts = Counter(codons)
    total = sum(counts.values())
    return {f"codon_{c}": counts.get(c, 0) / total if total > 0 else 0
            for c in all_codons}

#Apply to data frame and add a feature columns with codon frequencies 
codon_features = df["cds_sequence"].apply(codon_frequencies)
codon_df = pd.DataFrame(codon_features.tolist())

df = pd.concat([df, codon_df], axis=1)


#Define a codon table mapping codons to their respective amino acids
CODON_TABLE = {
    "TTT":"F","TTC":"F","TTA":"L","TTG":"L",
    "CTT":"L","CTC":"L","CTA":"L","CTG":"L",
    "ATT":"I","ATC":"I","ATA":"I","ATG":"M",
    "GTT":"V","GTC":"V","GTA":"V","GTG":"V",
    "TCT":"S","TCC":"S","TCA":"S","TCG":"S",
    "CCT":"P","CCC":"P","CCA":"P","CCG":"P",
    "ACT":"T","ACC":"T","ACA":"T","ACG":"T",
    "GCT":"A","GCC":"A","GCA":"A","GCG":"A",
    "TAT":"Y","TAC":"Y","TAA":"*","TAG":"*",
    "CAT":"H","CAC":"H","CAA":"Q","CAG":"Q",
    "AAT":"N","AAC":"N","AAA":"K","AAG":"K",
    "GAT":"D","GAC":"D","GAA":"E","GAG":"E",
    "TGT":"C","TGC":"C","TGA":"*","TGG":"W",
    "CGT":"R","CGC":"R","CGA":"R","CGG":"R",
    "AGT":"S","AGC":"S","AGA":"R","AGG":"R",
    "GGT":"G","GGC":"G","GGA":"G","GGG":"G"
}

aa_columns = sorted(set(CODON_TABLE.values()) - {"*"})
aa_df = pd.DataFrame(0, index=codon_df.index, columns=[f"aa_{aa}" for aa in aa_columns])

# Sum codon frequencies per amino acid
for codon, aa in CODON_TABLE.items():
    if aa == "*":  # skip stop codons
        continue
    codon_col = f"codon_{codon}"
    if codon_col in codon_df.columns:
        aa_df[f"aa_{aa}"] += codon_df[codon_col]

#Combine the data frames

combined_df = pd.concat([df["Name"], codon_df, aa_df], axis=1)

# Reset index
combined_df = combined_df.reset_index(drop=True)

# Save to CSV file
combined_df.to_csv("codon_aa_frequency.csv", index=False)

print("Combined features saved to 'codon_aa_frequency.csv'")

