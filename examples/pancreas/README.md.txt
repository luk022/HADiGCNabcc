Pancreas Example

This folder contains a full reproduction of the validation experiment using pancreas data (Baron dataset as training and Segerstolpe dataset as testing) used in the thesis.

Files

• example_pancreas_full.py
Main script for running the experiment.
It contains preprocessing, feature selection, adjacency processing, dataset creation, model training and evaluation.

• dataset/
Folder where all input CSV, NPZ and NPY files must be placed.

• results/
Output folder generated automatically. It will contain training logs, metrics and optional plots.

Running the Experiment

Run the script from the project root or from inside this folder

python example_pancreas_full.py


Ensure the dataset folder contains

baron_shared_expression_cleaned.csv
Baron_processed_labels.csv
segerstolpe_shared_expression_cleaned.csv
Segerstolpe_processed_labels.csv
processed_Baron_adjacency_scipy_format.npz
processed_Baron_adjacency_scipy_format_gene_names.npy


All preprocessing functions come directly from your original uploaded code.

Debug Version

To test the script quickly, reduce the runtime by modifying the following lines inside the script

num_epochs = 300     change to 3
global_seeds_list = [32, 42, 52]     change to [32]
DEFAULT_SORTER_LIST = ["odd_even", "softsort"]     change to ["softsort"]
SORTER_TO_PK["softsort"] = [[0.4, 0.4, 0.1, 0.1]]


This keeps only a single sorter, a single seed and a single Pk configuration so the script runs in seconds.


End of pancreas example README.