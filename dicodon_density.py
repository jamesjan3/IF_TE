import pandas as pd
import csv

INPUT_CSV = "TE_eIF_depletion.csv"
OUTPUT_CSV = "dicodon_density.csv"

INHIBITORY_PAIRS = {
    "AGGCGA","AGGCGG","ATACGA","ATACGG",
    "CGAATA","CGACCG","CGACGA","CGACGG","CGACTG","CGAGCG",
    "CTCATA","CTCCCG",
    "CTGATA","CTGCCG","CTGCGA",
    "CTTCTG",
    "GTACCG"
}

df = pd.read_csv(INPUT_CSV)
df = df.dropna().reset_index(drop=True)

with open(OUTPUT_CSV, "w", newline="") as out:
    writer = csv.writer(out)
    writer.writerow(["Name", "dicodon_count", "dicodon_density"])

    for idx, row in df.iterrows():
        name = row["Name"]

        seq = str(row["tx_sequence"]).upper().replace("U", "T")
        u5  = int(row["utr5_length"])   # ← FIXED

        if u5 >= len(seq):
            continue

        cds = seq[u5:]

        # Trim to full codons
        remainder = len(cds) % 3
        if remainder != 0:
            cds = cds[:-remainder]

        if len(cds) < 6:
            writer.writerow([name, 0, 0.0])
            continue

        codons = [cds[i:i+3] for i in range(0, len(cds), 3)]
        cds_codons = len(codons)

        n_skipped_dicodons = 0
        count = 0

        for i in range(cds_codons - 1):
            dicodon = codons[i] + codons[i+1]
            if "N" in dicodon:
                n_skipped_dicodons += 1
                continue
            if dicodon in INHIBITORY_PAIRS:
                count += 1

        possible_positions = (cds_codons - 1) - n_skipped_dicodons
        density = count / possible_positions if possible_positions > 0 else 0.0

        writer.writerow([name, count, round(density, 6)])

print(f"Results written to {OUTPUT_CSV}")
