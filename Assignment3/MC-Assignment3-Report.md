# DATA622 - Assignment 3

*Marco Castro*

## Support Vector Machines

### Introduction

This assignment extends Assignment 2 by incorporating Support Vector Machine (SVM) models to analyze our Portuguese bank marketing dataset. I will review literature comparing the SVM against Decision Tree-based models in an area of interest to gain a greater understanding of how these algorithms are typically implemented. I will then train SVM models and compare their results against the results of the previous Decision Trees, Random Forest, and Adaptive Boosting (AdaBoost) from Assignment 2 and use insights from the literature review to help determine which model might be most suitable to help predict if a client will subscribe to bank term deposit through a marketing campaign.

### Area of Interest

My area of expertise is in the domain of housing security, including homelessness intervention/prevention and housing affordability. In particular, I hope to understand how machine learning (ML) can be used to mitigate homelessness by using widely available datasets such as the Homeless Management Information System (HMIS), US Census, and Bureau of Labor Statistics (BLS) data to identify key markers of individuals at-risk of becoming homeless so as to deploy preventative interventions. A secondary goal would be increase the efficiency of service delivery so address funding gaps and capacity constraints.

A preliminary search for "Homelessness" + "SVM" + "Decision Tree" and "Homelessness" + "Machine Learning" and related terms (i.e. displacement, housing insecurity) yielded few relevant results; while some articles included the correct keywords in tables or their respective literature reviews, much of the body of work focused on implementation or policy recommendations and did not present a detailed discussion on SVM or Decision Tree-based algorithms. Some studies relied primarily on logistic regression or discussed one of the two algorithms of interest; other studies used SVM or Decision-Trees as part of an ensemble model or custom model. Finally, Showkat et al. (2023) conducted a critical analysis of 40 research studies on ML in homeless service provision. While their paper analyzed which themes other studies prioritized (i.e. novel ML approaches, program performance and service limitations) and which themes were de-prioritized (i.e. human-centered approaches), it's worth noting that they faced similar challenges in terms of a) the volume of work and b) the specific algorithms that previous work discussed. This suggests that SVMs are seldomly used in this domain, perhaps because these models are less interpretable than less complex models such as Decision Trees or Linear Regression.

Given the lack of viable results in my preliminary search, I expanded my definition to include proxies for housing insecurity, including social determinants (poor health or substance abuse: Tabar et al. 2020), migration (Liyanage, 2023) bad credit (Doko, 2021), and loan defaults. The literature review that follows focuses on the topic "loan defaults" under the premise that a person unable to pay back debt may be at higher risk of loosing their housing others. While this topic is somewhat removed from the original goal, it is an attempt to understand if methods used for related sectors can inform future work. In other words: can this information be leveraged to develop ML methods for identifying individuals that have defaulted on housing loans to help folks stay in their homes as a way to address one facet of the housing and homelessness crisis?

### Literature Review

To ground my understanding of Decision Trees and SVMs, I reviewed [Ahmad et al. (2021)](https://onlinelibrary.wiley.com/action/showCitFormats?doi=10.1155%2F2021%2F5550344) and [Guhathakurata et al. (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8137961/), two studies that used Machine Learning to predict COVID-19 infections.

For their study, Ahmad et al. evaluated various Decision Tree and ensembles including Random Forests, Boosted ensembles and Bagging, with additional focus on techniques to address class imbalance. They found that Random Undersampling (RUS)–based Balanced Random Forests outperformed models trained with SMOTE, likely because synthetic over-sampling introduced noise. RUSBoost models, however, performed poorly. Given the data's high class imbalance, the authors found that "Accuracy" was inadequate for measuring model performance and favored F-measure, precision, recall, AUROC (Area Under the Receiver Operating Characteristic Curve), and AUPRC (Area Under the Precision-Recall Curve). Recall, F1 score and AUPRC were deemed to be the best performance metrics once the data was balanced with RUS, with AUPRC also edging out AUROC for imbalanced data.

While Ahmad did not work with SVMs, Guhathakurata et al. placed special interest on SVMs' ability to use kernels to make non-linear problems separable by transforming low-dimensional data to higher-dimensional space. Using three classes, their best SVM model achieved an overall accuracy of 87%, with perfect recall for severely infected cases, suggesting strong sensitivity for the most critical class. Performance was moderate for mildly or non-infected cases likely due to data imbalance, as confirmed by F1-scores reflected. When compared against K-nearest neighbors (KNN), Naïve Bayes, AdaBoost, Random Forest and DT across metrics including ROC, accuracy, precision, recall, and F1-score, SVM consistently outperformed all other models.

Turning to the literature for my area of interest, I reviewed three works on loan defaults (\[Ali et al. (2021)(https://www.ijiemjournal.uns.ac.rs/images/journal/volume12/IJIEM_272.pdf), [Chen et al.(2021)](https://www.sciencedirect.com/science/article/abs/pii/S0377221720306846), and \[Krasovytskyi and Stavyskyy (2024)\](https://www.journals.vu.lt/ekonomika/article/download/34809/34141\])) and a complimentary article on credit risk [Doko et al. (2021)](https://www.mdpi.com/1911-8074/14/3/138).

Azhar et al. evaluated 19 machine learning models, including multiple SVMs, decision trees, homogeneous ensemble classifiers and neural networks. Artificial neural networks (ANNs) achieved the highest overall prediction accuracy (97%) on the test dataset, while Boosted (21%) and RUSBoost Trees (18%). This is consistent with Ahmad's findings that found RUSBoosted performed weakly. Among SVMs, the coarse Gaussian kernel performed best (89% accuracy), followed by medium Gaussian (87%), cubic (84%), and fine Gaussian (82%), matching Bagged Trees (82%) and outperforming fine, medium, and coarse decision tree variants which had accuracy scores of 78%, 61% and 74% respectively. The researchers also evaluated for prediction speed, where RUSBoost and Boosted Trees were able to predict the highest number of observations per second though with poor predictive performance. It should be noted that while ANN's were the third fastest model, making it the best model when looking at speed and accuracy. The decision tree models exhibited faster prediction speed over SVMs, making them a suitable candidate if speed and interpretability are important.

Chen compared a range of machine learning and statistical models using AUC, H-measure, and Brier Score to evaluate classification performance. Results showed that heterogeneous ensemble methods consistently performed best, followed by neural networks, XGBoost, generalized additive models (GAM), and advanced logistic models (GLM2 and GLM3). In contrast, AdaBoost and KNN under-performed, even when compared to the simplest model (GLM1). SVM ranked in the middle of the group. Despite performance declines across all models when using out-of-time testing, the heterogeneous ensembles remained leaders in AUC, while GLM2, GLM3, GAM, and stacked models led in H-measure. Random Forest, AdaBoost, SVM, and KNN consistently lagged behind. This suggests that the complexity of a model is not necessarily associated with its performance and that a simpler statistical models may be suitable for risk classification tasks.

In their study Krsovytskyi and Stavyskyy compared several classification algorithms, including Logit, LDA, SVM, Random Forest, CART, neural networks, and XGBoost but did not include DTs due to performance issues. To combat data imbalance, the authors tested two techniques: weighting and SMOTE. RUS was not tested. When evaluating with AUROC, SVM and LDA performed the worst on the weighted dataset with a prediction power close to random selection (0.5); their performance improved somewhat after when using SMOTE. Random Forest and XGBoost achieved the best results, with XGBoost outperforming others overall.

Krsovytskyi and Stavyskyy build off of work by Dokos work on credit risk assessment. Doko’s study compared DTs, Random Forests, SVMs, neural networks, Logistic Regression (LR) models. As the data was imbalanced, Doko tested feature scaling and SMOTE. With feature scaling, Decision Trees performed best, followed by Random Forest, LR, SVM, and neural networks; without scaling, model performance declined overall. Applying SMOTE unexpectedly worsened performance, likely due to overfitting from synthetic samples. This point is similar to findings by Ahmad where SMOTE introduced noise and did not perform as well as RUS.

Several points stood out when reviewing these studies to consider while continuing this project. In particular: \* Accuracy is generally unreliable for imbalanced datasets. Alternative metrics such as F1 score, recall, AUROC, and AUPRC provide a more informative assessment of model performance. In our marketing context, where the cost of a “yes” response may outweigh that of a “no,” recall may remain the preferred metric despite the prevailing preference of F1 score in the studies. \* SMOTE can potentially introduce noise due to its synthetic sample generation. RUS may achieve less noisy outcomes but at the expense of potentially valuable datapoints. Thus SMOTE may still be preferable for smaller datasets or in cases with severe data imbalance, as undersampling to a minority class with few data points could weaken our predictors. \* SVM performance is context-dependent. It was the top-performing model for COVID-19 detection (Guhathakurata, 2021) but showed inconsistent results in loan default prediction: among the worst performers in Krsovytskyi and Stavyskyy (2024), mid-range in Chen (2021), and strong but behind ANN and KNN in Azhar et al., indicating that domain or data characteristics may strongly influence outcomes.

### SVM Modeling

A Support Vector Machine (SVM) is a supervised learning algorithm that finds an optimal "hyperplane" to separate classes. By using kernel functions, SVM can transform nonlinearly separable data into higher-dimensional spaces to enable linear separation.

#### Preprocessing and Data Balancing

The SVM models will use a pipeline for preprocessing, including imputation and one-hot encoding of categorical variables like our Random Forest and AdaBoost pipelines. Numerical values also be standardized in the pipeline for SVM to work properly. Since the minority class `yes` makes up only 11% of our data, I will have selected SMOTE over RUS as a data balancing method in order to prevent loosing valuable data points and insights that might influence our predictions.

#### Model Building

I began by building a baseline model (svm0_pipe) using the arbitrary values for penalty cost (C=0.1) and gamma='scale'. When using scale for gamma, it is automatically calculated to balance the effects of each feature. I chose 'RBF' as the kernel as EDA from Assignment 1 indicated that our data is not linear.

##### Is 'RBF' a good kernel?

To confirm that this was the case, I used the dimensionality reduction technique Principal Component Analysis (PCA) to visualize the data in 2 dimensions. However, Figure 1 shows visible banding as a results of the one-hot encoding of the categorical variables. A scatter-plot of t-SNE (Figure 2) -- which can work with a mix of numerical and categorical variables -- suggests that our data may be better suited for a 'RBF' kernel over a 'linear' kernel.

To test if RBF is correct and to fine tune our C and gamma hyper-parameter, I used 5-fold Cross-Validation to test 'RBF', 'poly' and 'linear' kernels, C values of 0.1, 1, and 10, and gamma values of 'scale', 0.1, 0.01. Table 1 shows the mean Accuracy, Recall, and F1-score for the 12 model variants. As the test dataset is very likely imbalanced and the literature indicated that accuracy was not an adequate metric for imbalanced data, I will focus on F1 Score and Recall. Figure 1 and 2 show line plots for F1 Score and Recall by C and gamma for mean F1 Score and Recall respectively.

Cross-validation confirms that using a linear kernel does not perform as strongly as our best 'RBF' models and shows little difference between C and gamma values. Surprisingly, using a linear kernel does not yield the worst performing model for either evaluation metric. The best Recall (57.97% using our training data) score was yielded when using the hyper-parameters kernel='rbf', C=0.1, and gamma=0.01, while the best F1 Score (43.4% using our training data) came from hyper-parameters kernel='rbf', C=10, and gamma=0.01. Models `svm1_pipe` and `svm2_pipe` reflect these values respectively.

##### Can Recall be improved by feature selection?

Given that the current models yield moderate Recall (57.97%) and F1 Score (43.4%), I wanted to explore if reducing model complexity through feature selection would mitigate possible noise introduced by SMOTE. First, I used the function permutation_importance to score each of my model's features. Table 2 shows a ranking of these features from most important to least. Next I filtered out any feature with an importance mean less than 0.005. This reduced the number of features from 40 to 14. I trained a new model (`svm3_pipe`) using the smaller feature set and performed Cross-Validation to tune the hyper-parameters as before. Model `svm3_pipe` uses an RBF kernel, C=0.1, and gamma=0.01 and yielded a Recall of 56.53% and an F1 score of 33.02% on our training data.

##### Perfomance on Test Data

When evaluating with our test dataset, model `svm2_pipe` had the best accuracy (0.859) and F1 Score (0.456). However, model `svm1_pipe` had the best Recall value of 0.576868. Table 1 (below) shows the values of the evaluation metrics for the three SVM models along with the values for all models from Assignment 2.

**Table 1: Comparison of All Models**

| Model Name | Accuracy | Recall (Yes) | F1 (Yes) | AUC | AUPRC |
|------------|------------|------------|------------|------------|------------|
| SVM 1 | 0.804726 | 0.576868 | 0.399602 | 0.760356 | 0.380418 |
| SVM 2 | 0.858784 | 0.525862 | **0.456217** | 0.762382 | 0.384800 |
| SVM 3 | 0.819859 | 0.478448 | 0.374368 | 0.721687 | 0.326938 |
| ------------- | ----------- | --------------- | ----------- | -------- | -------- |
| DT Base | 0.830460 | 0.312500 | 0.293423 | 0.605826 | 0.174159 |
| DT 1 | 0.848345 | 0.504310 | 0.428310 | 0.752236 | 0.348789 |
| DT 2 | 0.859675 | 0.466236 | 0.428100 | 0.727157 | 0.333665 |
| DT 3 | 0.528688 | **0.778017** | 0.271089 | 0.680707 | 0.240591 |
| ------------- | ----------- | --------------- | ----------- | -------- | -------- |
| RF Base | 0.863235 | 0.368534 | 0.377761 | 0.735150 | 0.310198 |
| RF 1 | 0.869629 | 0.290948 | 0.334572 | 0.727504 | 0.309402 |
| RF 2 | 0.871409 | 0.308908 | 0.351164 | 0.732431 | 0.314691 |
| ------------- | ----------- | --------------- | ----------- | -------- | -------- |
| Ada Base | 0.888565 | 0.346983 | 0.412292 | 0.753653 | 0.383261 |
| Ada 1 | **0.891317** | 0.304598 | 0.387038 | **0.763200** | 0.389658 |
| Ada 2 | 0.890669 | 0.317529 | 0.395526 | 0.762900 | **0.393265** |
| Ada 3* | 0.879501 | 0.407328 | 0.432329 | 0.754521 | 0.383991 |
*Previously selected as best model

### Section 3: Model Comparison

All three SVM models performed well across the various performance metrics when compared to all previous models, as seen in Table 1. `DT3` had the highest recall, but was one of the worst performing models across all other metrics along with `DT Base`. All Decision Tree (`DTx`) and all Random Forest models (`RFx`) were generally outperformed by the SVM and AdaBosst models.  While `SVM 2` stood out with the highest F1 score and competitive scores for all other metrics, it was edged out to `Ada 1` for AUC and Accuracy and `Ada 2` for AUPRC. However, the three SVM models performed better than the AdaBoost models for Recall with scores ranging from 0.478 to 0.577 vs 0.305 to 0.407.  

If we were to follow the examples from the literature, we should put greater emphasis on F1 score, AUC, and AUPRC. If this is the case, our best model may lie between `SVM 2`, `Ada 1` and `Ada 2`. However, given that the goal of our analysis is to predict if a client will subscribe to bank term deposit, we could prioritize Recall, as every additional "yes" outcome translates to an additional client subscription leading to profits for the bank, while misidentifying a "no" outcome has a comparatively low cost. Our performance metrics thus suggest that `SVM 1` may have a fairly high success rate for predicting positive outcome. Finally, though `SVM 2` may slightly under-predict positive outcomes, it may produce predictions that better balance costs and benefits.

### Conclusion

The result of the tests suggest that model `SVM 1` may be our strongest model to generate the highest "yes" recall value while generally performing well across other metrics. However, model `SVM 2` should not be discounted as it produces the most balanced "yes" and "no", predictions leading to better balance between profits from clients that will sign up and the cost of targeting those that won't. While the results of these SVM models is encouraging, other factors may need to be explored before putting either of these models into production. For example, our literature review suggested that SVM models may be relatively slow to train, potentially increasing our computing costs, a factor that is of particular importance if we need to retrain our model frequently. Additionally, SVMs are not as interpretable as other types of models, a factor that could make it difficult to garner the same level of organizational buy-in as a simpler model. Finally, a greater understanding of this domain may be needed to ensure we are selecting an appropriate model.

### References

Ahmad, Amir, Safi, Ourooj, Malebary, Sharaf, Alesawi, Sami, Alkayal, Entisar, Decision Tree Ensembles to Predict Coronavirus Disease 2019 Infection: A Comparative Study, Complexity, 2021, 5550344, 8 pages, 2021. https://onlinelibrary.wiley.com/doi/10.1155/2021/5550344

Azhar Ali, S.E, Rizvi, S. S. H., F. Lai, F., Faizan Ali, R., and A. Ali Jan Predicting (2021) Delinquency on Mortgage Loans: An Exhaustive Parametric Comparison of Machine Learning Techniques International. Journal of Industrial Engineering and Management, 12 (1), 1. https://www.ijiemjournal.uns.ac.rs/images/journal/volume12/IJIEM_272.pdf

Chen, S., Guo, Z. and X. Zhao (2021). Predicting Mortgage Early Deliquency with Machine Learning Methods. European Journal of Operational Research, 290(1), 358. https://www.sciencedirect.com/science/article/abs/pii/S0377221720306846

Doko, F., Kalajdziski, S., & Mishkovski, I. (2021). Credit risk model based on Central Bank Credit Registry data. Journal of Risk and Financial Management, 14(3), 138. https://www.mdpi.com/1911-8074/14/3/138

Guhathakurata S, Kundu S, Chakraborty A, Banerjee JS. A novel approach to predict COVID-19 using support vector machine. Data Science for COVID-19. 2021:351–64. doi: 10.1016/B978-0-12-824536-1.00014-9. Epub 2021 May 21. [MCID: PMC813796P](https://pmc.ncbi.nlm.nih.gov/articles/PMC8137961/)

Krasovytskyi, D. adn A. Stavyskyy (2024). Predicting Mortgage Loan Defaults Using Machine Learning Techniques. Predicting Mortgage Loan Defaults Using Machine Learning Techniques. Ekonomika, 103(2). https://www.journals.vu.lt/ekonomika/article/download/34809/34141

#### Additional Sources

Bastos, J. A. (2022). Predicting Credit Scores with Boosted Decision Trees. Forecasting, 4(4), 925-935. https://doi.org/10.3390/forecast4040050

Liyanage, C. R., Mago, V., Schiff, R., Ranta, K., Park, A., Lovato-Day, K., Agnor, E., & Gokani, R. (2023). Understanding Why Many People Experiencing Homelessness Reported Migrating to a Small Canadian City: Machine Learning Approach With Augmented Data. JMIR formative research, 7, e43511. https://pmc.ncbi.nlm.nih.gov/articles/PMC10189624/

Showkat, D., Smith, A. D. R., Wang, L., and A. To. (2023, April). "Who is the right holmess client?": Values in ALgorithmic Homelessness Service Provision and Machine Learning Reasearch \[Conference presentation\]. 2023 ACM Conference on Human Factors in Computing Systems (CHI). Hamburg, Germany. https://www.researchgate.net/publication/370131660_Who_is_the_right_homeless_client_Values_in_Algorithmic_Homelessness_Service_Provision_and_Machine_Learning_Research

Tabar, M., Park, H., Winkler, S., Lee. D., Barman-Adhikari, A. and A. Yadav, (2020, August 23-27). Identifying Homeless Youth At-Risk of Substance Use Disorder: Data-Driven Insights for Policymakers \[Conference presentation\]. KDD'20 Virtual Event. USA. https://pike.psu.edu/publications/kdd20-homeless.pdf

Tan, Jialu ()Using machine learning to identify populations at high risk for eviction as an indicator of homelessness \[Masters Thesis, Massachusetts Institute of Technology\]. https://dspace.mit.edu/handle/1721.1/127660