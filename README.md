Bagging of XGBoost Classifiers with Random Under-sampling and Tomek Link for Noisy Label-imbalanced Data

Application of Ruisen et al: 
https://www.researchgate.net/publication/328005489_Bagging_of_Xgboost_Classifiers_with_Random_Under-sampling_and_Tomek_Link_for_Noisy_Label-imbalanced_Data

Dataset from Kaggle:
https://www.kaggle.com/sakshigoyal7/credit-card-customers

Class imbalanced data presents consistent challenges to accurate classification models. This approach incorporates random undersampling, tomek link elimination, 
and bagging of xgboost models to achieve state-of-the-art performance on noisy, imbalanced datasets.

Here I use an example dataset from Kaggle to compare Ruisen to a baseline xgboost model to predict credit card churn.

I find that Ruisen achieves about the same ROC AUC score (99%+ on this dataset), and better Recall, but lower Precision compared to the baseline XGBoost model.

This is an expected result since the random undersampling and tomek link elimination are intended to improve our ability to correctly predict the minor class.

In summary, this is a useful model for applications with highly imbalanced data where the cost of false negatives outweigh the cost of false positives.
