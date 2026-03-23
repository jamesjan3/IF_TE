How to Run
1 Run the .py files in main to extract sequence features from the raw data in TE_eIF_depletion.csv (this takes up hundreds of MBs)
2 (optional) run check.py as a sanity check on all extracted .csv files
3 run merge.py to get the entire input dataframe
4 run your desired subset model.py in its directory

we did it this way because we each worked on feature extractions separately:
  length features (Jannes)
  Nucleotide frequency (Anuschka)
  Codon and aa frequency (Oliver)
  K-mer frequency (Jannes)
  Wobble position (Anuschka)
  one-hot encoding of nt around start codon (Jannes)
  Counts of 20 relevant dicodons according to Gamble et al (Jannes)
  Secondary structure values (Oliver)




Instructions for LGBM:

Running the LGBM.py file will create a folder with directories for each of the defined feature subsets along with a general summary of the model results.

Inside the feature subset directories are two folders with the model results for the two depletion conditions (eIF3d and eIF4e). They contain .csv files with the search results across the folds (currently 5), feature importance scores, and predicted vs observed values on the test set. A model summary is also given as a .json file.


Currently the feature subsets are the ones described in the report.

BEFORE RUNNING:
-You will need to put the merged_features.csv file into the LGBM directory after building it or specify the file path