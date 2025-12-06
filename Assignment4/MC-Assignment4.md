MC - Assignment #4
 
## Introduction 

Studies suggest that roughly 40 percent of students started but did not graduate higher education (). High dropout rates can have a negative impact both for the individual and businesses. Many individuals who dropout leave with debt. According the Federal Reserve, 28 percent of people who received some but never completed college education had student debt ([federalreserve.gov, 2025](https://www.federalreserve.gov/publications/2025-economic-well-being-of-us-households-in-2024-higher-education-and-student-loans.htm?utm_source=chatgpt.com)). Additionally, depending on the particular field, a college degree can serve as a proxy for revelant job skills and used to screen the applicant pool. A 2022 study by the Burning Glass Institute and Harvard Business School found that 37% of middle skilled jobs still require a college. This puts these individuals at a potential economic disadvantage, as some may find that they don't meet the expected qualifications in certain job markets while also having to worry about paying back their debt. This can make it difficult to find their way back to school or to gain upward mobility. While some employers are experimenting resetting/lowering their degree requirements ([Burning Glass Institute, 2022](https://static1.squarespace.com/static/6197797102be715f55c0e0a1/t/6202bda7f1ceee7b0e9b7e2f/1644346798760/The+Emerging+Degree+Reset+%2822.02%29Final.pdf)), those that still have degree requirements risk turning away highly skilled candidates that could restrict the growth potential of the business and its broader sector. 

This study aims to better identify factors that contribute to student dropouts. This information could be used to provide interventions for students at risk of dropping out such as recruiting these students into shorter skill-based programs designed to meet business demands in an effort to boost the economic outcomes for students and businesses alike. I will use a dataset from [Kaggle](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention/data) to analyze 34 variables covering topics such as student's personal profile (i.e. marital status, gender, age, debt), their school profile (i.e. enrollment type, major), academic achievement over two semesters (i.e. number of courses attempted and passed), parent's educational level and occupation, and local economic conditions (i.e. unemployment, inflation rate, and GPD). Using the Machine Learning algorithms Decision Trees, XGBoost and Neural Networks,  I will train evaluate models with the intent of identifying the strongest predictive model. The selected models will be used to ultimately answer: _what are the factors that contribute to dropouts?_

## 1. EDA and Data Engineering

The dataset is comprised of 4424 samples with 16 numerical variables, 18 categorical variables including our dependent variable `target`. Each sample represents a unique student record for one year (2 semesters) of college participation with an outcome (Dropout, Graduate and Enrolled) assigned to `target`. Roughly 18% of our observations were categorized as **Enrolled**, meaning that the student is still in school does not have a final outcome (graduated vs dropped out). For the purposes of this study, I have held out these observations to use as my inference set to predict the likely final status, as training my models with this class might make obscure the signals for predicting dropouts and graduates.  80% of the remaining 3467 observations will be used to train my models, while 20% will be used to test them.

|----------|------|-------|
| Graduate | 2129 | 50.31% |
| Dropout  | 1338 | 38.59% |

The dataset has no missing (N/A) values. However, 180 observations (~4%) had a value of 0 in all twelve fields for academic achievements in the last two semesters. These fields should include information about the number of courses attempted, passed, and credits awarded per semester, as well the corresponding number of tests and grades. Figure 1 shows the boxplots for our continuous variables demonstrating that students that dropout are more likely to have received lower grades and passed fewer courses than those that graduated or those still enrolled.  The records in question had mixed outcomes in the `target` field, where some were enrolled and others were graduates or dropouts making it difficult to identify a particular pattern for why this value was used. This suggests that 0 could be a stand-in for a missing value possibly from previous imputation, that the dataset was synthetically generated, or that these records could be representative of edge cases (ie students that completed/were enrolled in course work in a previous year but whose outcome status was not updated until the year of observation), as it seems highly unlikely that a student could be a graduate without taking at least one course. Given the connection between low grades and failed courses with dropouts and that we have no way of verifying these records, I opted to drop them over performing imputation as to avoid introducing noise into our model. 

Our boxplots also show that the IQRs for `unemployment_rate` and `application_order` are practically identical for the Dropout and Graduate classes, suggesting that these variables may not add much information to our models. I omited these two variables from my model so as to help reduce the overall dimensions used by my models and improve training efficiency. 

Examining our categorical variables show other areas where we may be able to bin categories to reduce dimensions created by One-Hot encoding. In particular, `application_mode`, `previous_qualifications`, mother and father's "qualification" and "occopation" fields, and `nationality` all have multiple categories of few observations (less than 5%) that can be grouped together to keep the number of dimensions from growing too high.

Testing for Pearson Correlation Coefficient and Variance Inflation Factor (VIF) shows high correlation between 1st and 2nd semester academic achievement fields as seen in Table 1 and Table 2. I created new engineered variables combining some of these fields to help minimize correlation and as a dimension reduction method. The fields created were:

* `total_enrolled` : the sum of `curricular_units_1st_sem_enrolled` and `curricular_units_2nd_sem_enrolled` represeing the total number of courses that the student was enrolled in during the school year;
* `total_credited`: the sum of `curricular_units_1st_sem_credited` and `curricular_units_2nd_sem_credited` representing the total number of credits received during the school year;
* `total_approved`: the sum of `curricular_units_1st_sem_approved` and `curricular_units_2nd_sem_approved` representing the total classes passed during the school year;
* `total_evaluations`: the sum of `curricular_units_1st_sem_evaluations` and `curricular_units_2nd_sem_evaluations` representing the total number of courses with tests during the school year;

Finally, I also created an engineered field for the students weighted average grade (`weighted_avg_grade`) and ommited the `curricular_units_[]num_sem_without_evaluations` fields as these fields are complimentary to the `curricular_units_[num]_sem_evaluations` fields and thus contain reduntant information.

## 3.  Model Selection

I used three different Machine Learning techniques to evaluate their predictive performance for Accuracy, Precision, Recall, F1 Score, AUC, AUPRC across 6 models, as shown on Table 1. While all 6 metrics were of importance, Accuracy was less important than Recall for my analysis, given that our goal is to capture the highest number of at-risk students F1 Score is also important as a high F1 Score it is the harmonic will balances precision and recall, ensuring that we prioritize both. 

I trained a Decision Tree with a max depth of 5 as my baseline model due to the high interpretability of Decision Trees. Although this performed decently in Accuracy and Precision, Decision Trees are also highly prone to high variance and overfitting since small changes in the training data can produce different splits. 

I proceeded by experiment with XGBoost, an ensemble learning method that builds many shallow trees sequentially to adjust errors resulting in a more accurate and efficient model with slightly higher bias but much lower variance than a single tree, thus preventing overfitting. After training an initial model using random hyperparameters (XG0), I manually tested different values for `max_depth`, `min_child_weight`, `gamma`, `n_estimators`, and `learning_rate` with 5-fold CV to create model `xg1`. I them proceeded to use Random Search to perform some hyperparemeter tuning using 10-fold Cross-Validation (CV) and applied the best results to `XG2`. All three models performed similarly with XG0 outperfoming models on Recal

For my final set of experiments, I trained Neural Networks (NN) models. A neural network is a machine learning model that learns patterns from data by passing information through layers of connected nodes. I tested Neural Networks with 1 and 2 hidden layers and compared different input and output dimensions. 

Table 1: Comparison of Performance Metrics Across All Models

| Model   | Accuracy |     Prec |   Recall |       F1 |      AUC |    AUPRC |
|:--------|---------:|---------:|---------:|---------:|---------:|---------:|
| DTree   | 0.880747 | **0.951456** | 0.728625 | 0.825263 | 0.892428 | 0.87016  |
| XG0     | 0.905172 | 0.931915 | **0.814126** | 0.869048 | 0.932589 | 0.936413 |
| XG1     | 0.903736 | 0.93913  | 0.802974 | 0.865731 | 0.938457 | 0.939775 |
| XG2     | 0.905172 | 0.939394 | 0.806691 | 0.868    | 0.938379 | 0.938942 |
| NN1     | 0.905172 | 0.935622 | 0.810409 | 0.868526 | **0.940599** | 0.940281 |
| NN2     | **0.906609** | 0.935897 | **0.814126** | **0.870775** | 0.939911 | **0.940774** |



Based on the table below, NN2 is our best performing model for Recall, F1 Score, AUC, and AUPRC, and performed relatively well for Precision and Accuracy. Given the nature of our study, NN2 


## Conclusion
Make your conclusions from your analysis. Please be sure to address the business impact (it could be of any domain) of your solution.