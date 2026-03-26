# Translational Efficiency Prediction

This project predicts the **translational efficiency (TE)** of genes from RNA sequence features using machine learning. It models log-transformed TE values under eIF3d and eIF4e depletion conditions with Ridge regression, ElasticNet, and LightGBM models. Features are extracted from transcript sequences and merged into a single dataset for training.

# How to Run

1. Unzip `TE_eIF_depletion.zip`
2. Run the `.py` files in main to extract sequence features from the raw data in `TE_eIF_depletion.csv` (this takes up hundreds of MBs)
3. *(optional)* Run `check.py` as a sanity check on all extracted `.csv` files
4. Run `merge.py` to get the entire input dataframe
5. Run your desired model in its directory

We did it this way because we each worked on feature extractions separately:

| Feature | Author |
|---------|--------|
| Length features | Jannes |
| Nucleotide frequency | Anuschka |
| Codon and aa frequency | Oliver |
| K-mer frequency | Jannes |
| Wobble position | Anuschka |
| One-hot encoding of nt around start codon | Jannes |
| Counts of 20 relevant dicodons according to Gamble et al | Jannes |
| Secondary structure values | Oliver |

---

## Instructions for LGBM

Running `LGBM.py` will create a folder with directories for each of the defined feature subsets along with a general summary of the model results.

Inside the feature subset directories are two folders with the model results for the two depletion conditions (eIF3d and eIF4e). They contain `.csv` files with the search results across the folds (currently 5), feature importance scores, and predicted vs observed values on the test set. A model summary is also given as a `.json` file.

Currently the feature subsets are the ones described in the report.
