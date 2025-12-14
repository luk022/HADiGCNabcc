# DLPFC Example (HADiGCN Thesis Replication)

This script reproduces the DLPFC experiments used in the thesis.  
It performs:

- 3-fold cross-validation on slice 151673  
- Testing on:  
  - Scenario 1: slice 151676  
  - Scenario 2: slice 151507  
- Baseline CE loss ("celoss")  
- Top-K losses ("odd_even", "softsort")  
- Evaluation using Top-1 accuracy, Top-3 accuracy, Micro F1, and Macro F1  

---

## Run Full Experiment

```bash
python example_dlpfc_full.py
```

This runs:

- full training  
- 3-fold cross-validation  
- CE baseline and all Top-K configurations  

---

## Debug Mode (3 epochs)

Use this mode to quickly test dataset loading and adjacency consistency:

```bash
python example_dlpfc_full.py --debug
```

Debug mode uses:

- 3 epochs  
- seed = [32]  
- only the "softsort" sorter  
- only one Pk: [0.4, 0.0, 0.6]  

This provides a fast way to verify that the script runs correctly.

---

## Sorter to Pk Configuration

```python
SORTER_TO_PK = {
    "celoss": [None],

    "odd_even": [
        [0.4, 0.0, 0.6],
        [0.7, 0.2, 0.1],
    ],

    "softsort": [
        [0.4, 0.0, 0.6],
        [0.8, 0.0, 0.2],
    ],
}
```

- "celoss" uses the standard cross-entropy loss  
- "odd_even" and "softsort" use the top-k loss implementation  

---

## Metrics Computed

For every fold, seed, sorter, and Pk:

- Top-1 accuracy  
- Top-3 accuracy  
- Micro F1  
- Macro F1  

---

## Notes

- The script assumes that the processed DLPFC slices and adjacency matrix are already provided.  
- No preprocessing script is required.  
- You may replace the adjacency matrix file later if a newer graph becomes available.  
