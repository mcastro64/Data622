MC - Assignment #4
 
## 0. Introduction 

Studies suggest that roughly 40 percent of students started but did not graduate higher education (). High dropout rates can have a negative impact both for the individual and bussinesses. Many individuals who dropout leave with debt. According the Federal Reserve, 28 percent of people who received some but never completed college education had student debt ([federalreserve.gov, 2025](https://www.federalreserve.gov/publications/2025-economic-well-being-of-us-households-in-2024-higher-education-and-student-loans.htm?utm_source=chatgpt.com)). Additionally, depending on the particular field, a college degree can serve as a proxy for revelant job skills and used to screen the applicant pool. A 2022 study by the Burning Glass Institute and Harvard Business School found that 37% of middle skilled jobs still require a college. This puts these individuals at a potential economic disadvantage, as some may find that they don't meet the expected qualifications in certain job markets while also having to worry about paying back their debt, making it difficult to find their way back to school or to gain upward mobility. While some employers are experimenting reseting their degree requirements ([Burning Glass Institute, 2022](https://static1.squarespace.com/static/6197797102be715f55c0e0a1/t/6202bda7f1ceee7b0e9b7e2f/1644346798760/The+Emerging+Degree+Reset+%2822.02%29Final.pdf)), those that still have degree requirements risk turning away highly skilled candidates that could  restrict the growth potential of the business and its broader sector. 

This study aims to better identify factors that contribute to student dropouts. This information could be used to provide interventions for students at risk of dropping out such as recruiting them into shorter skill-based accredited programs designed in collaboration with businesses in an effort to boost the economic outcomes for students and businesses alike. I will use a dataset from [Kaggle](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention/data) to analyze 34 variables covering topics such as student's personal profile (i.e. marital status, gender, age, debt), their school profile (i.e. enrollment type, major), academic achievement over two semesters (i.e. number of courses attempted and passed), parent's educational level and occupation, and local economic conditions (i.e. unemployment, inflation rate, and GPD). Using the Machine Learning algorithms Random Forest, XGBoost, Support Vector Machines and Neural Networks,  I will train evaluate models with the intent of identifying the strongest predictive model. The selected models will be used to ultimately answer: what are the factors that contribute to dropouts?

## 1. Data

The dataset is comprised of 4424 samples with 16 numerical variables, 18 categorical variables including our dependent variable `target`. Each sample represents a unique student record for one year (2 semesters) of college participation with an outcome (Dropout, Graduate and Enrolled) assigned to `target`. Roughly 18% of our observations were categorized as "enrolled" meaning that the student is still in school does not have a final outcome (graduated vs dropped out). Keeping this class to train my models might make obscure the signals for predicting dropouts and graduates. For the purposes of this study, I have held out these oservations to use as my inference set to predict the likely final status. 80% of the remaining 3467 observations will be used to train my models, while 20% will be used to test them.

|----------|------|-------|
| Graduate | 2129 | 50.31% |
| Dropout  | 1338 | 38.59% |

The dataset has no missing (N/A) values. However, 180 observations (~4%) had a value of 0 in all twelve fields for academic achievements in the last two semesters. These fields should include information about the number of courses attempted, passed, and credits awarded per semester, as well the corresponding number of tests and grades. Figure 1 shows the boxplots for our continous variables demonstrating that students that dropout are more likely to have received lower grades and passed fewer courses than those that graduated or those still enrolled.  The records in question had mixed outcomes in the `target` field, where some were enrolled and others were graduates or dropouts making it difficult to identify a particular pattern for why this value was used. This suggests that 0 could be a stand-in for a missing value possibly from previous imputation, that the dataset was synthetically generated, or that these records could be representative of edge cases (ie students that completed/were enrolled in coursework in a previous year but whose outcome status was not updated until the year of observation), as it seems highly unlikely that a student could be a graduate without taking at least one course. Given the connection between low grades and failed courses with dropouts and that we have no way of verifying these records, I opted to drop them over performing imputation as to avoid introducing noise into our model. 

Our boxplots also show that the IQRs for `unemployment_rate` and `application_order` are practically identical for the Dropout and Graduate classes, suggesting that these variables may not add much information to our models. I omited these two variables from my model so as to help reduce the overall dimensions used by my models and improve training efficiency. 

Examining our categorical variables show other areas where we may be able to bin categories to reduce dimensions created by One-Hot encoding. In particular, `application_mode`, `previous_qualifications`, mother and father's "qualification" and "occopation" fields, and `nationality` all have multiple categories of few observations (less than 5%) that can be grouped together to keep the number of dimensions from growing too high.

Testing for Pearson Correlation Coefficient and Variance Inflation Factor (VIF) shows high correlation between 1st and 2nd semester academic achievement fields as seen in Table 1 and Table 2. I created new engineered variables combining some of these fields to help minimize correlation and as a dimension reduction method. The fields created were:

* `total_enrolled` : the sum of `curricular_units_1st_sem_enrolled` and `curricular_units_2nd_sem_enrolled` represeing the total number of courses that the student was enrolled in during the school year;
* `total_credited`: the sum of `curricular_units_1st_sem_credited` and `curricular_units_2nd_sem_credited` representing the total number of credits received during the school year;
* `total_approved`: the sum of `curricular_units_1st_sem_approved` and `curricular_units_2nd_sem_approved` representing the total classes passed during the school year;
* `total_evaluations`: the sum of `curricular_units_1st_sem_evaluations` and `curricular_units_2nd_sem_evaluations` representing the total number of courses with tests during the school year;

Finally, I also created an engineered field for the students weighted average grade (`weighted_avg_grade`) and ommited the `curricular_units_[]num_sem_without_evaluations` fields as these fields are complimentary to the `curricular_units_[num]_sem_evaluations` fields and thus contain reduntant information.

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