Assignment

Choose a dataset
You get to decide which dataset you want to work on. The data set must be different from the ones used in previous homeworks You can work on a problem from your job, or something you are interested in. You may also obtain a dataset from sites such as Kaggle, Data.Gov, Census Bureau, USGS or other open data portals. 
Select one of the methodologies studied in weeks 1-10, and another methodology from weeks 11-15 to apply in the new dataset selected.

To complete this task:. 
Describe the problem you are trying to solve.
Describe your dataset and what you did to prepare the data for analysis. 
Methodologies you used for analyzing the data
What's the purpose of the analysis performed
Make your conclusions from your analysis. Please be sure to address the business impact (it could be of any domain) of your solution.

Deliverable

The traditional R file or Python file and essay,
An Essay (minimum 500 word document) or Video (5 to 8 minutes recording)
Include the execution and explanation of your code. The video can be recorded on any platform of your choice (Youtube, Free Cam).


Problem Statement:
What are the factors that contribute to dropouts?

## 2. Feature Engineering

Our dataset has no missing (N/A) values. However, 180 of 4424 (~4%) observations have not values entered into any of the 12 1st and 2nd semester fields fields, but had different outcomes/targets. I will drop since this indicates that the student was not in school during the period of interest and records could just add noise to our models.


A handful of rows (11 total) had credited > approved, but it seems questionable that you can get credit w.out passing class. Likely data entry error.

18% of our observations are marked as in "enrolled", which is the student's current state. Keeping this class might make it harder to find the signals for dropout as students my drop out later. Therfore we will drop it, even as that means shrinking our dataset. We should watch out for curse of dimensionality and may need to apply Dimensionality reduction techniques.
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
