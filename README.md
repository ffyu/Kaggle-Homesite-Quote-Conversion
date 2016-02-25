# Kaggle-Homesite-Quote-Conversion
Python code for 31st solution

I've tried Vowpal Wabbit, XGBoost, and Neural Networks with dropouts for this competition.

XGBoost has the best performance, followed by Neural Networks with a large margin.

The final submission is the average of three XGBoost models. The feature engineering makes the key impact here. The first model used one-hot-encoding for non-numeric features. The second and third models both performed the out-of-fold numeric encoding for all the categorical features.
