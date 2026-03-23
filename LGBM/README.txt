Instructions for LGBM:

Running the LGBM.py file will create a folder with directories for each of the defined feature subsets along with a general summary of the model results.

Inside the feature subset directories are two folders with the model results for the two depletion conditions (eIF3d and eIF4e). They contain .csv files with the search results across the folds (currently 5), feature importance scores, and predicted vs observed values on the test set. A model summary is also given as a .json file.


Currently the feature subsets are the ones described in the report.

BEFORE RUNNING:
-You will need to put the merged_features.csv file into this directory after building it