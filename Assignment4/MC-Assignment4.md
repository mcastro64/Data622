MC - Assignment #4
 
## 0. Introduction and Methodology

Describe the problem you are trying to solve.
Describe your dataset and what you did to prepare the data for analysis. 
Methodologies you used for analyzing the data
What's the purpose of the analysis performed


Debt. Loss of income potential.

Business end:
- College degree as a screener.
- Loss of possible good candidates

What are the factors that contribute to dropouts?

## 1. Data

[Kaggle](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention/data)

The dataset is comprised of 4424 samples with 16 numerical variables, 18 categorical variables including our dependent variable `target`. Each sample represents a unique student record for one year (2 semesters) of college participation with an outcome (Dropout, Graduate and Enrolled) assigned to `target`. The supporting variables cover topics such as student's personal profile (i.e. marital status, gender, age, debt), their school profile (i.e. enrollment type, major), academic achievement over two semesters (i.e. number of courses attempted and completed), parent's educational level and occupation, and local economic conditions (i.e. unemployment, inflation rate, and GPD)

The dataset has no missing (N/A) values. However, 180 observations (~4%) had a value of 0 in all twelve academic achievement fields. These records had mixed outcomes in the `target` field where some were enrolled and others were graduates or dropouts; therefore, 0 appears to be a substitute for a missing value, as it appears  

I will drop since this indicates that the student was not in school during the period of interest and records could just add noise to our models.

A handful of rows (11 total) had credited > approved, but it seems questionable that you can get credit w.out passing class. Likely data entry error.

18% of our observations are marked as in "enrolled", which is the student's current state. Keeping this class might make it harder to find the signals for dropout as students my drop out later. Therfore we will drop it, even as that means shrinking our dataset. We should watch out for curse of dimensionality and may need to apply Dimensionality reduction techniques.

Reduced some correlation but still see highly correlated variables among "curriculum". Trees don’t care about correlated features but some SVM and NN

Our data shows some imbalance. Ratio is closer if we do not count enrolled.

Graduate   2129       50.31
Dropout    1338       31.62
Enrolled    765       18.08
Percent dropped out (if enrolled not counted):
38.59244303432362


#### Grouping
The following section groups similar categories together to assist with dimensionality reduction.


## 3. Model Training
### DT
Max depth = 7

### RF
n_estimator converges on 400 for both `gini` and `entropy` for 10-Fold Cross Validation
use RandomSearch to tune `max_depth`, `sample_split`, `samples_leaf`, `max_features`

### XGBoost

### SVM
With and without smote
### Neural Networks

## 4. Model Evalution

Table 1: Comparison of Performance Metrics Across All Models
| Model            | Accuracy   | Precision    | Recall       | F1         | AUC          | AUPRC       |
|------------------|------------|--------------|--------------|------------|--------------|-------------|
| Decision Tree    | 0.877522   | 0.892704     | 0.776119     | 0.830339   | 0.878210     | 0.858240    |
| RF0              | 0.902017   | 0.938596     | 0.798507     | 0.862903   | 0.940776     | 0.940898    |
| RF1              | 0.903458   | 0.938865     | 0.802239     | 0.865191   | **0.943531** | 0.942258    |
| XG0              | 0.914986   | 0.940928     | 0.832090     | 0.883168   | 0.941656     | 0.943156    |
| XG1              | 0.909222   | 0.936170     | 0.820896     | 0.874751   | 0.943066     | 0.943740    |
| XG2              | 0.909222   | 0.936170     | 0.820896     | 0.874751   | 0.943066     | 0.943740    |
| SVM0             | 0.904899   | 0.942982     | 0.802239     | 0.866935   | 0.941656     | 0.941852    |
| SVMlinear        | 0.907781   | **0.943478** | 0.809701     | 0.871486   | 0.935661     | 0.938823    |
| SVMlin w.SMOTE   | 0.891931   | 0.887550     | 0.824627     | 0.854932   | 0.932240     | 0.935464    |
| SVM_RBF          | 0.900576   | 0.919831     | 0.813433     | 0.863366   | 0.938122     | 0.939636    |
| SVM_RBF w.SMOTE  | 0.894813   | 0.904564     | 0.813433     | 0.856582   | 0.938126     | 0.938646    |
| NN1              | 0.906340   | 0.907631     | **0.843284** | 0.874275   | 0.936322     | 0.940285    |
| NN5              | **0.916427** | 0.937500   | 0.839552   | **0.885827** | 0.939642     | **0.943420** |

## Conclusion
Make your conclusions from your analysis. Please be sure to address the business impact (it could be of any domain) of your solution.