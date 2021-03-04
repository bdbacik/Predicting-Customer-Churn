## Predicting Bank Customer Churn Using Machine Learning: Project Overview

### Problem Statement 
* A manager at the bank is disturbed with more and more customers leaving their credit card services. They would really appreciate if one could predict for them which customers they can expect to churn so they can proactively intervene and offer services and products to retain the customer, and ultimately achieve a positive return on investment for the bank.

### Project Summary
* Built classification model to predict churn of credit card customers for a bank.
* Optimized Linear Regression, Random Forest, and XGBoost with grid search cross validation.
* Introduced novel modeling approach with random undersampling, tomek link elimination, and xgboost bagging to to achieve better positive class Recall on noisy, label imbalanced data.
* Conducted ROI analyis by considering customer lifetime value (LTV), cost of false positives/negatives, and cost of intervention to retain customers.  
* Presented recommendation for final model and summarized learnings for similar problems. 

### Code and Resources Used
Python Version: 3.7
Packages: pandas, numpy, sklearn, xgboost, matplotlib, seaborn

### Data Description
* Dataset obtained from Kaggle: https://www.kaggle.com/sakshigoyal7/credit-card-customers
* This dataset consists of 10,000 customers with 21 features describing customer age, salary, marital_status, credit card limit, credit card category, etc.  The data describes customer behavior from the past 12 months. 
* 16.1% of customers have churned during this time period. Thus, it's a good example of an imbalanced class problem that will require careful modeling to achieve strong performance.  

### Exploratory Data Analysis
I looked at the distributions of the data and the value counts for the various categorical variables. Below are a few highlights from the analysis.

#### Correlation heatmap:
* The three most highly correlated variables with the response are 'Total_Revolving_Bal', 'Total_Trans_Ct', and 'Total_Ct_Chng_Q4_Q1'. Each of these predictors have negative correlations > 0.2 with the response variable 'y'.

![correlation heatmap](https://github.com/bdbacik/XGBoost-Applications/blob/main/images/corr_heatmap.png)

#### Boxplots and KDE plots:
* Next I plotted boxplots and kernel density plots for these three predictors to get a better idea of the distribution and relationship with the response.
* Notice a large number of outliers for 'Total_Trans_Ct' and 'Total_Ct_Chng_Q4_Q1'
* For 'Total_Revolving_Balance' we see add odd distribution, with the majority of churners centered around $0 balance.  

![boxplots and kdeplots](https://github.com/bdbacik/XGBoost-Applications/blob/main/images/boxplots.png)


### Data Preprocessing
I completed the following pre-processing steps...
* Handle missing values
* Convert categorical features to dummies
* Normalize/scale features using robust scaler to maintain outlier relationships without skewing distribution
* Split into training and test sets with 75/25 split
* Oversample minor class in training set to balance classes

### Model Selection and evaluation

I tried three different models and evaluated them using the F1 Score metric. The F1 score can be interpreted as a weighted average of the precision and recall, where an F1 score reaches its best value at 1 and worst score at 0. The relative contribution of precision and recall to the F1 score are equal. The formula for the F1 score is:

> F1 = 2 * (precision * recall) / (precision + recall)

F1 is particularly relevant for this problem because there is a clear business impact in the trade off between precision and recall.  We'll be able to use these metrics to conduct a ROI analysis.

I tried three different models:

* Logistic Regression – Baseline for the model
* Random Forest – decision tree ensemble method, non-parametric method may perform better due to complex relationship between predictors and outcome.
* XGBoost – gradient boosting technique is an additive (ensemble) technique that builds one tree at a time, learning from previous iterations.  Has achieved benchmark results in similar applications. 

### Results

* Logistic Regression: 0.855
* Random Forest: 0.962
* XGBoost: 0.974
* XGBoost with Random Undersampling, Tomek Link Elimination, and Model Bagging: 0.962


### Discussion

Now we can evaluate the business impact of our models by considering the lifetime value of a customer along with the cost of false negatives and false positives.  Ultimately we will select the model which achieves the greatest business ROI for the bank. 

* LTV (lifetime value of a customer) = total annual transaction amount * margin per transaction * retention time period
  * total annual transaction amount for churner = $3095
  * margin per transaction = 3%
  * retention time period = 3 years
* Thus, the average LTV for a churner = 3095 * 0.03 * 3 = $278.5

![ltv](https://github.com/bdbacik/XGBoost-Applications/blob/main/images/ltv.png)

Next, we need to quantify the cost of false positives and false negatives.
* False Positive (FP) = $100 (assumed cost of an incentive offer to customer to prevent churn)
* False Negative (FN) = $278.5 (the lifetime value of the customer)

Finally, we need the false positive and false negative rates for each model.  
 
![results](https://github.com/bdbacik/XGBoost-Applications/blob/main/images/results.png)

Now, we can calculate the expected value per prediction (EvP) in comparing our more advanced models to our baseline logistic regression model.
* EvP = FP cost * change in FP % + FN cost * change in FN %
  * For our xgboost model, EvP = $100 * 10.4% + $278.5 * 1.4% = $14.6 savings per customer
  * For the xgboost+ model, EvP = $100 * 8.3% + 278.5 * 2.3% = $15.2 savings per customer
* We see that our XGBoost model with random undersampling, tomek link elimination, and model bagging ultimately saves the bank more money by identifying a greater percentage of true positives (i.e. churner customers) even while resulting in more false positives that cost the bank money. 

This is an expected result since the random undersampling and tomek link elimination are intended to improve our ability to correctly predict the minor class.  This is a useful approach in any class imbalanced problem where the cost of false negatives outweighs the cost of false positives.
