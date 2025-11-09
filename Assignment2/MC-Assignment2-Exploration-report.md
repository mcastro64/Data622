# DATA622 - Assignment 2
_Marco Castro_

## Experimentation & Model Training

### Introduction

For Assignment 2, we were asked to conduct model training using a dataset from Portuguese bank conducting a marketing campaign. The primary goal of our overall analysis is to use machine learning to help predict if a client will subscribe to bank term deposit. For this second step of our project, we will focus on experimenting with models using Decision Trees, Random Forest, and Adapative Boosting (AdaBoost). 

---

### Section 1: Working with Decision Trees

A Decision Tree is a supervised learning model that predicts outcomes by recursively splitting data into branches based on feature values, forming a tree-like structure of decision rules. It is a highly interpretable model that is easy to compute, but prone to overfitting issues.

Before building the first Decision Tree model, I performed manual preprocessing, including feature selection, binning and one-hot encoding of categorical variables, addressing data imbalance and splitting my data in to testing and training datasets and testing datasets. I then generated a basic model that utilizes the default settings for the DecisionTreeClassifier. This model ("base model") will be used as a baseline for comparing against all other Decision Tree models generated in our experiments. The base model has an accuracy score of 83%. However, a closer look at the Confusion Matrix shows that while our model correctly predicted "no" responses 89.6%, only predicted "yes" responses correctly 31.3% of the time. Furthermore, the model considered all 48 features as important, making the tree overly complex; with low bias and high variance, this model has a high risk of overfitting and sensitivity to noise.

#### Q1.1: Can we simplify the model?
For my first experiment (Q1.1), I choose the arbitrary value of 0.03 as the Decision Tree stopping criterion ("min_impurity_decrease") which controls how much the impurity must decrease before the model can make a split. Even with a small stopping criterion, model DT1 reduced the number of important features to 11. The models' overall accuracy drops to 84.83% as the model is no longer fitting exactly to the training set, thus increasing the models bias and reducing its variance relative to our base model and preventing overfitting. We also see a nearly 20% increase in predicting "yes" responses from 31.3% to 50% over the base model, while predicted "no" responses saw only a small drop of 0.3% (89.2%). 

#### Q1.2: Will post-pruning improve our model's accuracy?

Next, I was curious if implementing post-pruning would the predictive power of the model (Q1.2). I used a loop to iterate through different possible Cost Complexity Pruning (CCP) alpha values with which to model. Figure 1.1 shows that the model has the best best accuracy on the training dataset between ~0.006 and 0.015. However, the model's accuracy on the training dataset began a steady declining around 0.003. Both our training and test datasets performed well between 0 and ~0.003. 

#### Q1.3: Does our best CCP value range generalize well?

While our model's accuracy scored highest between ~0.006 and 0.015, I wanted to know if these values were due to chance or if the CCP alpha values generalized well (Q1.3a). I used 10-Fold Cross-Validation to examine the results. Figure 1.2 shows that the model performs best with CCP alpha values smaller than ~0.003. To find a precise best CCP alpha value and stopping criterion values, I used GridSearch with 5-Fold Cross Validation. This resulted in a CCP alpha of 0.003271 (Q1.3b) and stopping criterion of 0.005556 (Q1.3c). I set as to their respective DecisionTreeClassifier parameters to generate Model DT2. Model DT2 saw a small overall accuracy improvement over model DT1 (85.96% vs 84.83% respectively) and higher correctly predicted "no" answers (91%) than our base or DT1 models. While DT2 performed better than the base model at correctly predicting "yes" response, the model saw a decrease of 3% compared to DT1 (47%). Like model DT1, DT2 has a better bias and variance balance than our base model.

#### Q1.4: How does the absence of seasonality affect our model

Model DT2 contains identified several of the months as important (Jul, May, Aug, Nov, and April), while discarding the rest of the months in our dataset. For the final Decision Tree experiment, I wanted to see how removing `months` as a predictor would affect the models accuracy by setting the maximum depth to 4 (Q1.4). Model DT3 saw an improvement of correctly predicted "yes" responses (78%) but at the cost of correctly predicted "no" respones (50%) and overall accuracy (52.87%). While this might be easiest model to interpret, our model appears to be overly simplified and underfits the data and exhibits high bias and lower variance when compared to the DT1 and DT2. 

---

### Section 2: Working with Random Forest 

A Random Forest is an ensemble machine learning technique that combines the predictions from many decision trees to improve overall performance. Compared to a single Decision Tree, it generally offers higher accuracy, greater stability, and reduced overfitting by averaging out individual tree errors.

#### Q2.1: How does the choice of criterion affect our Model?

For experiment Q2.1, I wanted to explore there significant differences in accuracy between two criterion: "gini" and "entropy". The "criterion" in the RandomForestClassifier is a parameter used to adjust how the algorithm makes a split a branch when building a Decision Tree. By default, the RandomForrestClassifier uses Gini Impurity, a metric that measures the probability of incorrectly classifying a random sample from the node. While the gini impurity is stated as achieving good results, it can also produce overly optimistic accuracy results when compared to alternate criterion, such as entropy, measures the amount of uncertainty in the node which can produce more balanced splits than gini impurity. Although both criterion often produce similar results, models using the entropy criterion can sometimes achieve accuracy results that are somewhat more realistic when comparing the training and testing datasets over using gini impurity but at a higher computational cost. This experiment seeks to note those differences to understand if it may justify the added computational cost. 

Figure 2.1 shows a comparison of number of trees (50-500) in our Random Forest vs the model accuracy. Both algorithms have similar Out-of-Bag (OOB) accuracy scores across all number of trees, with differences between 0.000020 (350) and 0.000723 (100 trees). Entropy produced the highest overall accuracy (92.5204% when using 500 trees). The highest accuracy for Gini impurity was 92.5126% when using 200 trees. Both criterion produced nearly identical accuracy when using 350 trees (92.5087% and 92.5107%). These very small differences don't provide much evidence that we should pick one criterion over the other. We will therefore continue to use the default Gini impurity for our models.

It should be noted we may be able to achieve good results even when choosing a smaller number of trees in our model. For example, models using entropy performed comparatively well with an accuracy of 92.5048% when using only 100 trees and 92.5165% when using 300 trees. This suggests that we may be able to trade accuracy for to improve our computational performance over the best performing entropy model that achieved 92.5204% accuracy when using 500 trees, but we would need to perform speed tests in order to compare against our Gini impurity models and thus have the added cost of testing. Conversely, increasing the number of trees reduces variance and results in more stable predictions.


#### Q2.2: Does Cross-Validation confirm the reliability of our Gini scores?

Given the high predictive values of the previous test, I was curious if I could confirm the accuracy values of the Gini impurity models with Cross-Validation. Cross-Validation would also help understand how well the model would generalize and validate the number of trees that I should use in my model. A Cross Validation Mean Score test using 5-Folds CV shows that our models are within .0025 and .0054 of the accuracy scores from Q2.1 as seen on Figure 2.2. The 5-Fold CV test also shows that the model performed best when using 350 trees (91.98%). 

Given that this value also performed well for both our Gini Impurity and Entropy models, I built a baseline model (RF0) using n_estimator=350 and a Gini Impurity for its faster computational performance. Model RF0 produced an overall accuracy of 86.32%, with a recall value of 93% for "no" responses and 37% for "yes" responses.

#### Q2.3: Does using a pipeline for preprocessing instead of manual preprocessing reduce the difference between our Gini Values and our Cross-Validation accuracy values?

While researching Random Forests, I stumbled across the following recommendation:

> “Using a pipeline helps prevent data leakage by ensuring 
> that the same transformations are applied to the test data 
> as were applied to the training data, without reusing 
> information from the test data in the training process.”
— [Scikit-learn User Guide: Pipeline and Composite Estimators]
(https://scikit-learn.org/stable/modules/compose.html#pipeline)

After building a pipeline, I used n_estimators from 50-500 to test accuracy scores for Random Forest models using Gini impurity OOB scores and cross validation as before. The results of this test yielded similar slightly higher differences between the Gini OOB and our CV Score (0.00398 to 0.005489). Model RF1 uses the pipeline to update our previous model and has overall accuracy of 87.10% with a recall of 94% for "no" responses and 31% for "yes" responses. Despite the decline in predicted "yes" responses, I will keep using the pipeline in subsequent tests to avoid data leakage.

#### Q2.4: How does selecting different subsampling affect model accuracy?

For the final test, I want to test how the choice of sub-sampling affects my model accuracy. Using a five possible options for max_feature ('sqrt', 'log2', 0.3, 0.5, None]), we see that the model using `log2` just edges out models using `sqrt`, with OOB scores of 0.9307 and 0.9305 respectively. Using `log2` can introduce more randomness between trees and thus lessen correlation and reduce the risk of overfitting. Conversely, using `sqrt` can produce a better balance between bias and variance. Model RF2 uses `sqrt` for max_features and produces similar results as RF1 with a slight improvement in overall accuracy (87.14%).

---

### Section 3: Working with AdaBoost 

AdaBoost is another ensemble method that sequentially combines multiple weak learners by giving more weight to misclassified samples in each round to improve overall accuracy. Unlike a single Decision Tree or a Random Forest, which build trees independently, AdaBoost focuses on correcting errors from previous models, emphasizing hard-to-classify cases rather than averaging many trees. For our experiments, AdaBoost will be using many shallow decision trees as its weak learners / estimators.

Before developing a base model, I used GridSearch to calculate the best number of trees to use for the AdaBoost model (500). Model Ada0 resulted in an 89.17% overall accuracy score and a 0.96 recall value for "no" responses, but had similarly low recall values (0.33) as the Random Forest models for "yes" responses.

#### Q3.1: How does the choice of tree depth affect the model's accuracy? 

AdaBoost models typically use a shallow decision tree. For this experiment, I wanted to know if using a stump -- trees width a depth of 1 -- would be sufficient in the model.  This was tested by performing 5-Fold Cross Validation across models using trees of depths between 1-6. Figure 3.1 shows that accuracy increases as tree depth increases from 0.9103 at tree depth 1 to 0.9192 at tree depth 6. However, increasing the tree depth can have low bias and high variance that could overfit the model to the training data. Given that the change between our shallowest and deepest model is relatively small (under 0.1%), we would gain a very small improvement in accuracy but risk introducing lower bias and higher variance. It would, therefore be preferable, to keep our model simpler and use a tree depth of 1 or 2. For comparison purposes, Model Ada1 uses a tree depth of 2. Model Ada1 resulted in an 89.13% overall accuracy score similar to Ada0. The model was slightly better at predicting "no" responses (0.97) but did worse at predicting "yes" responses (0.30). 

#### Q3.2: How does the learning rate affect the accuracy of the model? 

For my final AdaBoost experiment, I wanted to find the optimal learning rate that would lead to the highest model accuracy. The learning rate in AdaBoost controls how much each weak learner contributes to the final ensemble prediction. A smaller learning rate reduces each learners’ impact, requiring more trees to achieve the same performance, while a larger rate increases each learners’ influence but can risk overfitting Models Ada0 and Ada1 used the default learning rate of 1.0. Similar to the experiment in Q3.1, I used 5-Fold Cross Validation across models using learning rates from 0.01 to 2.0. Figure 3.2 shows a curve with  an initial jump from and accuracy of 0.75 to 0.85 from a learning rate of 0.02 to 0.04 which begins to plateau around a learning rate of 0.5. Our best learning rate is 0.9166 when the learning rate is 1.79. However, a learning rate of 0.7495 produces a similar score (0.9118), while our default learning  rate resulted in accuracy of 0.91090. 

Using a high learning rate can lower model bias but can result in higher variance risks overfitting, while using a lower learning rate can result in lower variance and less sensitive to noisy data but may have higher bias if the model doesn't have enough trees and risk underfitting. Since previous tests suggested using 500 trees, the risk of underfitting may be relatively small. Model Ada2 with a tree depth of 2 uses a learning rate of 0.75 in attempt to provide a slightly more stable model than the default rate through lower variance but keeping bias relatively low. For comparison, I also generated model Ada3 that uses the same learning rate but a tree stump (depth 1). Ada2 has an accuracy of 0.8906 vs Ada3's accuracy of 0.8795. The recall value for "no" responses for Ada2 was 0.96 vs 0.94 for Ada3. However Ada3 had a higher recall value for "yes" responses (0.41) over Ada2 (0.31).

## Evaluating All Models

The table below summarizes the accuracy and recall scores for the different Decision Tree, Random Forest and AdaBoost models created these experiments. Based on these results, the AdaBoost "Ada3" is likely the best choice for predicting if a client will subscribe to bank term deposit. While other AdaBoost models yielded higher overall accuracy values, this model was better at predicting "yes" responses. For the specific use case, missing a potential "yes" outcome translates to lost revenue from loosing out on a customer signing for the bank service. Conversely, misclassifying a "no" outcome has a relatively lower cost (ie the time to contact the customer). 

The Random Forest and AdaBoost base models may also be suitable for our purposes; of the two Ada Base may edge out RF Base due to its higher overall predictive power. On the other end of the spectrum, DT3 is the least desirable model, despite having the best yes recall score, as this model had the worst overall prediction accuracy and may be overfitting the "yes" response.


| Model Name | Accuracy | Recall: No | Recall Yes |
|-------------|-----------|------------|-------------|
| DT Base     | 0.830460  | 0.896215   | 0.312500    |
| DT 1        | 0.848345  | 0.892020   | 0.504310    |
| DT 2        | 0.859675  | 0.909622   | 0.466236    |
| DT 3        | 0.528688  | 0.497036   | 0.778017    |
| RF Base     | 0.863235  | 0.926037   | 0.368534    |
| RF 1        | 0.871004  | 0.942544   | 0.307471    |
| RF 2        | 0.871409  | 0.942818   | 0.308908    |
| Ada Base    | 0.891721  | 0.962699   | 0.332615    |
| Ada 1       | 0.891317  | 0.965800   | 0.304598    |
| Ada 2       | 0.890669  | 0.963429   | 0.317529    |
| Ada 3       | 0.879501  | 0.939444   | 0.407328    |


### Conclusion and Next Steps

The results of several tests suggest that the Ada 3 may be our strongest model as it produces the high "yes" recall value while generally performing well. Given the context of this analysis, correctly identifying a potential signup should be a priority for our model. We should, however, be cautions to ensure that our model is not overfitting to our "yes" responses. We may want to consider our base AdaBoost as an alternative model and perform additional tests to confirm the validity of our selection. For example, when splitting our dataframe, we could reserve a subset of data for validation purposes, perform 10-fold Cross Validation instead of 5-fold CV, or use bootstrapping to test if our model is well generalized. Future work could also include adjusting our decision threshold. This may require further industry knowledge. We may also need fine tune our model for optimal outcomes and consider alternate ensemble methods and machine learning algorithms.