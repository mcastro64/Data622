# DATA622 - Assignment 3
_Marco Castro_

## Support Vector Machines

### Introduction

This assigment extends the previous work for Assignment 2 by incorporating Support Vector Machine (SVM) models to analyse our Portuguese bank marketing dataset. I will review literature comparing the SVM against Dection Tree-based models in an area of interest to gain a greater understanding of how these algorithms are typically implemented.  I will then train SVM models and compare their results against the results of the previous Decision Trees, Random Forest, and Adapative Boosting (AdaBoost) from Assigment 2 to and use  insights from the literature review to help determine which model might be most suitable to help predict if a client will subscribe to bank term deposit through a marketing campaign. 

### Area of Interest

My area of expertise is in the domain of housing security, including homelessness prevention + intervention and affordability. A prelimanary search for "Homelessness" + "SVM" +  "Decision Tree" and "Homelessness" + "Machine Learning" and related terms (ie.  displacement, housing insecurity) yielded few relevant results; while some articles included the correct keywords in tables or their respective literature reviews, much of the body of work focused on implementation of policies or recommendations and did not present a detailed discussion on SVM or Decision Tree-based algorithms. Some studies relied primarily on logistic regression or discussed one of the two algorithms of interest; other studies used SVM or Decision-Trees as part of an ensemble model or custom model. Finally, it's worth noting that one article reviewed 40 studies based on their use of machine learning and determined that ... This suggests that SVMs are seldomly used in this domain, perhaps because these models are less interpretable than Decision Trees models.

Given our poor results in my prelimenary attempt, I expanded the search to include additional terms such as credit risk, and other

### Literature Review 

1. Student demonstrates two (2) articles provided were read by, for example, drawing insights, summarizing articles or via comparison (5)
2. Three (3) articles provided with URL links (5)
3. Discussion of a) two articles provided, and b) three articles the students found (5)
4. Comparison and insight drawn for a) two articles provided, and b) three articles the students found (5)

Area of expertise/ interest	20	
1. Explanation of area of expertise/interest (10)
2. Results related to area of expertise/interest (10)

#### Area of Interest


Ahmad - Random Undersampling (RUS) Balanced random forest performed best over SMOTE, perhaps due to noise introduced in synthetic samples. Discusses how acuracy is not the best measure for imbalanced data;   F-measure, precision, recall, area under the precision-recall curve, and area under the receiver operating characteristic curve performance measures are employed to compare the performances of different decision tree ensembles. For initial test, ensemble of 50 trees; ration of .5; 200 trees best 
preference for AUROC and AUPRC . Age important
As the data is imbalanced, classification accuracy is not an appropriate performance measure to compare different classifiers. preference for F1:  
 Also uses AUROC and AUPRC for comparison; preference for AUROC:
 "It has been demonstrated [37] that, for imbalanced datasets, AUPRC is a better performance measure as compared to AUROC." 

 "Rus on majority class with a ratio of .5 did best.

 "The results are consistent with the theory of classifier ensembles that most of the performance improvement comes with the first few classifiers in ensembles, adding more classifiers may not be very useful"
 Doesn't really compare SVM

Guhathakurata - SVM performed best:
"SVM is chosen because it uses kernel trick to convert low-dimensional input space to high-dimensional space and thus converts the nonseparable problem to separable problem."
C=10
" We are looking for a smaller margin hyper-plane to classify the infected classes more accurately with fewer miss-predictions;"
"model has an accuracy of 87%. From the classification report, we see that our methodology has a high success rate of predicting severely infected case"
" The classifier has shown perfect result for predicting the cases which belong to severely infected class since it has recall = 1 but has shown a moderate result for the other two classes. As an f1-score, which is the average of precision and recall, is high for severely infected and a bit low for not infected and mildly infected."

"Our analysis on COVID-19 dataset depicts that among all the other supervised models, SVM works best in predicting COVID-19 cases with maximum accuracy. "

"SVM outperforms all other models that are tested [kNN, Naïve Bayes, RF, AdaBoost, Binary Tree, and SVM] [using] ... ROC, CA, F1 Score, Precision, and Recall. The confusion matrix for all the models has been summarized in Fig. 18.7 , illustrating the superiority of SVM in predicting COVID-19 "

Krsovytskyi - Compared Logit, LDA, SVM, RF, CART, NNET, XG8TREE. SVM and LDA showed worst perfomance (very similar to random guess) on imbalanced but weighted data (did not predict any defaults), did a little better after SMOTE. Logit overpredicted defaults (real 564, predicted 3000+)/ RF and XGBTree did the best. Most models liked GDP as highest IV, but not XGBTree(RF did not when weighting but did when SMOTE). XGBTree liked DSTI. No simple DT.

Doko  -  DT > RF > LR > SVM > NN for imbalanced data with scaling. did surprisingly worst with SMOTE with balancing (so not considered) and not as good without scaling. Perhaps this has to do with overfitting? Delayed Days best IV. Age not important.

Delinquency - 19 ML tested including upport
Vector Machine (SVM) [34] method was used with
four variants: coarse gaussian 89 > medium gaussian 87 >  cubic SVM 84 > fine gaussian SVM 82. Decision Tree was used with
three variants: fine tree 78 > coarse tree 74 >  medium
tree 61. For ensembled classifiers, we investigate five ho-
mogenous ensemble methods– RUSBoosted Trees,
subspace KNN, subspace discriminant, bagged trees,
and boosted trees. ANN performed best, boosted trees and RUSBoosted Trees performed worst, though in speed ensemble trees were faster. SVM did better on test data than DT, but did similar for training set (though maybe overfitting?). Training time and prediction time better for DT than SVM RUSBoosted Trees fastest but not accurate.

Age not important.

Chen - Use of AUC amd H-measure and Brier Score  to compare models. hetero-geneous ensemble methods lead other methods, followed by NN,
XGB, GAM, GLM3, and GLM2. ADA and KNN under-perform even
GLM1. SVM in the middle of the pack. For out-of-sample. the
performance difference between HC, Stack, and NN in the out-
of-sample data is rather minor. ADA and KNN still under-perform
GLM1 in the out-of-sample data. From the result using out-of-time data from 2016, we can see
that both the AUCs and H-measures decline noticeably from the
training and out-of-sample data, but the two heterogeneous meth-
ods still lead other methods based on the AUC performance mea-
sure, while GLM2, GLM3, GAM and Stack lead based on the H-
measure. However, the differences in either AUC or the H-measure
among GLM2, GLM3, GAM, XGB, NN, and the two heterogeneous
methods are not substantial, and all seven methods lead the re-
maining methods in the out-of-time data. ADA and KNN again
clearly trail the simple GLM1 based on both measures. Ranking:  RF, ADA, SVM, and KNN largely trail other methods. Conclusion: some ML methods, especially RF, ADA, SVM, and KNN, do not outperform better than some simpler methods, such as GLM2, GLM3, and GAM in risk classification.


### Section 2: SVM Model Training

Perform an analysis of the dataset used in Homework #2 using the SVM algorithm.

### Section 3: Model Comparison 

Compare the results with the results from previous homework.
Answer questions, such as:
Which algorithm is recommended to get more accurate results?
Is it better for classification or regression scenarios?
Do you agree with the recommendations?
Why?

__Table 1: Comparison of All Models

| Model Name | Accuracy | Recall (Yes) | F1 (Yes) |
|-------------|-----------|---------------|----------|
| DT 1        | 0.848345  | 0.504310      | 0.428310 |
| DT 2        | 0.859675  | 0.466236      | 0.428100 |
| RF 1        | 0.869629  | 0.290948      | 0.334572 |
| RF 2        | 0.871409  | 0.308908      | 0.351164 |
| Ada Base    | 0.888565  | 0.346983      | 0.412292 |
| Ada 1       | 0.891317  | 0.304598      | 0.387038 |
| Ada 2       | 0.890669  | 0.317529      | 0.395526 |
| Ada 3*      | 0.879501  | 0.407328      | 0.432329 |

*Previously selected as best model

### Section 4: Conclusion and Recommendations