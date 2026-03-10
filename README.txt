How to Run
1 Run the .py files in main to extract sequence features from the raw data in TE_eIF_depletion.csv (this takes up hundreds of MBs)
2 (optional) run check.py as a sanity check on all extracted .csv files
3 run merge.py to get the entire input dataframe
4 run your desired subset model.py in its directory

we did it this way because we each worked on feature extractions separately:
  length features (Jannes)
  Nucleotide frequency (Anushka)
  Codon and aa frequency (Oliver)
  K-mer frequency (Jannes)
  Wobble position (Anushka)
  one-hot encoding of nt around start codon (Jannes)
  Counts of 20 relevant dicodons according to Gamble et al (Jannes)
  Secondary structure values (Oliver)
