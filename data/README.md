\# Data



PhysioNet/CinC Challenge 2019, v1.0.0 — open access, CC BY 4.0

https://physionet.org/content/challenge-2019/1.0.0/



The data files are NOT stored in git (see `.gitignore`); only these README files are.

Each team member downloads the data once, from the project folder, with the `sepsis` environment active:



```

pip install awscli

aws s3 sync --no-sign-request s3://physionet-open/challenge-2019/1.0.0/training/ data\\raw\\training\\

```



Expected structure:



```

data/raw/training/

├── training\_setA/   hospital A, 20 336 files (p000001.psv, ...)

└── training\_setB/   hospital B, 20 000 files (p100001.psv, ...)

```



Each .psv file is one ICU patient: one row per hour, 41 columns separated by "|"

(8 vital signs, 26 laboratory values, 6 demographic/administrative variables, SepsisLabel).

