# csx46-cerenkov


This repository is for code from a final project in CSX46 at Oregon State University.

Methods and data come from Yao et al's work on the CERENKOV project, which uses XGBoost classifiers
to distinguish rSNP from non-regulatory cSNPs.

To replicate our results, simply run runTests.py with `python3 runTests.py`. This may take a very long time to run,
depending on the value of `nreps` and `k` in the file. Libraries used are Pandas, PyTorch, NumPy, imbalanced-learn, 
xgboost, scikit-learn, and matplotlib.

Our final poster with our results from this project can be viewed [here.](https://drive.google.com/file/d/116NPsWZndfbVuNo1EU8Qussdk29fB44g/view?usp=sharing)
If the above link isn't working, the final poster can also be downloaded from this repository, under 'Dibble_Fey_CSX46_Poster.pptx'.

Our work was based on/inspired by previous work on CERENKOV, which can be found at the [cerenkov](https://github.com/ramseylab/cerenkov) and [cerenkov3](https://github.com/ramseylab/cerenkov3) repositories.
