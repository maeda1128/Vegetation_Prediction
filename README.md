# Repository Overview

This repository contains Python scripts used for analyzing and comparing model performance in river channel characteristics using logistic regression models. Below is a brief description of the scripts included:

## `kfold_all_c.py`

This script calculates the model parameters and p-values based on cross-validation results specifically for Model C, which employs a logistic regression model.

## `kfold_*1_*2.py`

This script is designed to compare the accuracy of models A through D across different variables for each river segment using logistic regression. The naming convention is as follows:

- `*1` denotes the flood condition (`F`) or normal condition (`N`).
- `*2` represents the river segment, with options including `seg1`, `seg2_1`, or `seg2_2`.

## Dependencies

The following versions of Python and packages were used:

- **Python version:** 3.9.12
- **pandas version:** 1.5.2
- **seaborn version:** 0.11.2
- **statsmodels version:** 0.13.2
- **numpy version:** 1.23.4
- **matplotlib version:** 3.5.1
- **scikit-learn version:** 1.4.2

## Data

If you require access to the dataset used in these analyses, please contact me for further information.
