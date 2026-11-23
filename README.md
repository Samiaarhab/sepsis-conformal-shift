# sepsis-conformal-shift

## Data

PhysioNet/CinC Challenge 2019 v1.0.0 (open access, CC BY 4.0), not stored in git.
Download it from the project folder, with the `sepsis` environment active:

```
pip install awscli
aws s3 sync --no-sign-request s3://physionet-open/challenge-2019/1.0.0/training/ data\raw\training\
```

Expected: 20 336 files in `data/raw/training/training_setA` and 20 000 in `training_setB`.