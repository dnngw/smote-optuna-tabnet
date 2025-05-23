# The Effect of SMOTE and Optuna Hyperparameter Optimization on TabNet Performance for Heart Disease Classification

This study tries to conduct an experiment by applying SMOTE and Optuna optimization when performing classification using the TabNet deep learning algorithm. The heart disease dataset is used as a test sample to see the effects of SMOTE and Optuna on classification performance and can be more optimal than previous studies.

## Requirements
* pytorch-tabnet 4.1.0
* optuna 4.3.0
* pandas 2.2.2
* matplotlib 3.10.0
* seaborn 0.13.2
* numpy 2.0.2
* scikit-learn 1.6.1
* torch 2.6.0+cu124
* imbalanced-learn 0.13.0

All depedencies already in main.ipynb

## dataset
we use public dataset from kaggle.com

## Training
main.ipynb is the main program in the experiment flow. t_test_variance.ipynb is used to perform statistical calculations of the loss obtained during training. methodology, results, and explanation of the experiment have been described in the paper https://doi.org/10.32736/sisfokom.v14i2.2348
