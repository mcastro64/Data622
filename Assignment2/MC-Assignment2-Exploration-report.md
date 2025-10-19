# DATA622 - Assignment 1
_Marco Castro_

## Experimentation & Model Training

For Assignment 2, we were asked to conduct model training using a dataset from Portuguese bank conducting a marketing campaign. The primary goal of our overall analysis is to use machine learning to help predict if a client will subscribe to bank term deposit. For this second step of our project, we will focus on experimenting with models using Decision Trees, Random Forest, and Adapative Boosting (AdaBoost). 

## Section 1: Working with Decision Trees

I began by performing manual preprocessing, including feature selection, binning and one-hot encoding of categorical variables, addressing data imbalance and splitting my data in to testing and training datasets and testing datasets before generating a basic model that utilizes the default settings for the DecisionTreeClassifier. This basic model will be used as a baseline for comparing againts all other Decision Tree models generated in our experiments. The base model has an accuracy score of 83%. However, a closer look at the Confusion Matrix shows that while our model correctly predicted "no" responses 89.6%, only predicted "yes" responses correctly 31.3% of the time. Furthermore, the model considered all 48 features as important, making the tree overly complex.

### Q1.1: Can we simplify the model?
For my first experiment (Q1.1), I choose the arbitrary value of 0.03 as the Decision Tree stoping criterion ("min_impurity_decrease") which controls how much the impurity must decrease before the model can make a split. Even with a small stopping criterion, model DT1 reduced the number of important features to 11. As expected, the models' overall accuracy drops to 84.83% as the model is no longer fitting to exactly to the training set, thus preventing overfitting. We also see a small gain in predicting "yes" responses (50%), while "no" responses continue to be correctly predicted 89.2%.

### Q1.2: Will post-pruning improve our model's accuracy?
Next, I was curious if implementing post-pruning would the predictive power of the model (Q1.2). I used a loop to iterate through different possible Cost Complexity Pruning (CCP) alpha values with which to model. Chart 1 shows that the model has the best best accuracy on the training dataset between ~0.006 and 0.015. However, the model's accuracy on the training dataset began a steady declining around 0.003. Both our training our test datasets between 0 and ~0.003.

### Q1.3: Does our best CCP value range generalize well?
While our model's accuracy did well between ~0.006 and 0.015, I wanted to know if these values were due to chance or if the CCP alpha values generalized well (Q1.3a). I used 10 k-Fold Cross-Validation to examine the results. Chart 1.2 shows that the model performs best with CCP alpha values smaller than ~0.003. To find a precise best CCP alpha value and stopping criterion values, I used GridSearch with 5 k-Fold Cross Validation. This resulted in a CCP alpha of 0.003271 (Q1.3b) and stopping criterion of 0.005556 (Q1.3c) set as values to generate Model DT2. Model D2 saw small overall accuracy improvement over model DT1 (85.96% vs 84.83% respectively) and a gain in correctly predicted "no" answert (91%), but saw a drop in correctly predicted "yes" responses (47%).

### Q1.4: How does the absence of seasonality affect our model
Model DT2 contains indentified several of the months as important (Jul, May, Aug, Nov, and April), while discarding the rest of the months in our dataset. For my final Decision Tree experiment, I wanted to see how removing `months` as a predictor would affect the models accuracy (Q1.4). Model DT3 saw an improvement of correctly predicted "yes" responses (78%) but at the cost of correctly predicted "no" respones (50%) and overall accuracy (52.87%). 

## Random Forest


## AdaBoost 