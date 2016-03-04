# Kaggle-Homesite-Quote-Conversion
https://www.kaggle.com/c/homesite-quote-conversion

### Final private leader board ranking: 31/1764 

I've tried Vowpal Wabbit, XGBoost, and Neural Networks with dropouts for this competition.

XGBoost has the best performance, followed by Neural Networks with a large margin.

The final submission is the average of three XGBoost models. The feature engineering makes the key impact here. The first model used one-hot-encoding for non-numeric features. The second and third models both performed the out-of-fold numeric encoding for all the categorical features.

To run the models, please download the four data files from the link in **Data_Source_Link.txt** and then put them in the same folder. Under the Final_Submission.py, change the global variable **FOLDER**  to the directory that you've just put all the downloaded files. It might take a while to run due to the large size of the data (2 hours on an AWS c4.8xlarge instance).

Comments/suggestions are welcomed.

Best,

Feifei Yu
 
