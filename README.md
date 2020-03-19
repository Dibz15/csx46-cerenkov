# csx46-cerenkov


This repository is for code from a final project in CSX46 at Oregon State University.

Methods and data come from Yao et al's work on the CERENKOV project, which uses XGBoost classifiers
to distinguish rSNP from non-regulatory cSNPs.

To replicate our results, simply run runTests.py with `python3 runTests.py`. This may take a very long time to run,
depending on the value of `nreps` and `k` in the file. Libraries used are Pandas, PyTorch, NumPy, imbalanced-learn, 
xgboost, scikit-learn, and matplotlib.
