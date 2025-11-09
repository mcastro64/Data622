# DATA622 - Assignment 3
_Marco Castro_

## Support Vector Machines

### Introduction

This assignment extends Assignment 2 by incorporating Support Vector Machine (SVM) models to analyse our Portuguese bank marketing dataset. I will review literature comparing the SVM against Dection Tree-based models in an area of interest to gain a greater understanding of how these algorithms are typically implemented.  I will then train SVM models and compare their results against the results of the previous Decision Trees, Random Forest, and Adapative Boosting (AdaBoost) from Assigment 2 and use insights from the literature review to help determine which model might be most suitable to help predict if a client will subscribe to bank term deposit through a marketing campaign. 

### Area of Interest

My area of expertise is in the domain of housing security, including homelessness intervention/prevention and housing affordability. In particular, I hope to understand how machine learning (ML) can be used to mitigate homelessness by using widely available datasets such as the Homeless Management Information System (HMIS), US Census, and Bureau of Labor Statistics (BLS) data to identify key markers of individuals at-risk of becoming homeless so as to deploy preventative interventions. A secondary goal would be increase the efficiency of service delivery so address funding gaps and capacity constraints.

A preliminary search for "Homelessness" + "SVM" + "Decision Tree" and "Homelessness" + "Machine Learning" and related terms (ie.  displacement, housing insecurity) yielded few relevant results; while some articles included the correct keywords in tables or their respective literature reviews, much of the body of work focused on implementation or policy recommendations and did not present a detailed discussion on SVM or Decision Tree-based algorithms. Some studies relied primarily on logistic regression or discussed one of the two algorithms of interest; other studies used SVM or Decision-Trees as part of an ensemble model or custom model. Finally, Showkat et al. (2023) conducted a critical analysis of 40 research studies on ML in homeless service provision. While their paper analyzed  which themes other studies prioritized (ie novel ML approaches, program performance and service limitations) and which themes were deprioritized (ie. human-centered approaches), it's worth noting that they faced similar challenges in terms of a) the volume of work and b) the specific algorithms that previous work discussed. This suggests that SVMs are seldomly used in this domain, perhaps because these models are less interpretable than less complex models such as Decision Trees or Linear Regression.

Given the lack of viable results in my preliminary search, I expanded my definition to include proxies for housing insecurity, including social determinants (poor health or substance abuse: Tabar et al. 2020), migration (Liyanage, 2023) bad credit (Doko, 2021), and loan defaults. The literature review that follows focuses on the topic "loan defaults" under the premise that a person unable to pay back debt may be at higher risk of loosing their housing others. While this topic is somewhat removed from the original goal, it is an attempt to understand if methods used for related sectors can inform future work. In other words: can this information be leveraged to develop ML methods for identifying individuals that have defaulted on housing loans to help folks stay in their homes as a way to address one facet of the housing and homelessness crisis?

### Literature Review 

To ground my understanding of Decision Trees and SVMs, I reviewed [Ahmad et al. (2021)](https://onlinelibrary.wiley.com/action/showCitFormats?doi=10.1155%2F2021%2F5550344) and [Guhathakurata et al. (2021)](https://pmc.ncbi.nlm.nih.gov/articles/PMC8137961/), two studies that used Machine Learning to predict COVID-19 infections.

For their study, Ahmad et al. evaluated various Decision Tree and ensembles including Random Forests, Boosted ensembles and Bagging, with additional focus on techniques to address class imbalance. They found that Random Undersampling (RUS)–based Balanced Random Forests outperformed models trained with SMOTE, likely because synthetic over-sampling introduced noise. RUSBoost models, however, performed poorly. Given the data's high class imbalance, the authors found that "Accuracy" was inadequate for measuring model performance and favored F-measure, precision, recall, AUROC (Area Under the Receiver Operating Characteristic Curve), and AUPRC (Area Under the Precision-Recall Curve). Recall, F1 score and AUPRC were deemed to be the best performance metrics once the data was balanced with RUS, with AUPRC also edging out AUROC for imbalanced data. 

While Ahmad did not work with SVMs, Guhathakurata  et al. placed special interest on SVMs' ability to use kernels to make nonlinearly problems separable by transforming low-dimensional data to higher-dimensional space. Using three classes, their best SVM model achieved an overall accuracy of 87%, with perfect recall for severely infected cases, suggesting strong sensitivity for the most critical class. Performance was moderate for mildly or non-infected cases likely due to data imbalance, as confirmed by F1-scores reflected. When compared against K-nearest neighbors (KNN), Naïve Bayes, AdaBoost, Random Forest and DT across metrics including ROC, accuracy, precision, recall, and F1-score, SVM consistently outperformed all other models.

Turning to the literature for our area of interest, I reviewed three works on loan defaults ([Ali et al. (2021)(https://www.ijiemjournal.uns.ac.rs/images/journal/volume12/IJIEM_272.pdf), [Chen et al.(2021)](https://www.sciencedirect.com/science/article/abs/pii/S0377221720306846), and [Krasovytskyi and Stavyskyy (2024)](https://www.journals.vu.lt/ekonomika/article/download/34809/34141])) and a complimentary article on credit risk [Doko et al. (2021)](https://www.mdpi.com/1911-8074/14/3/138).

Azhar et al. evaluated 19 machine learning models, including multiple SVMs, decision trees, homogeneous ensemble classifiers and neural networks. Artificial neural networks (ANNs) achieved the highest overall prediction accuracy (97%) on the test dataset, while Boosted (21%) and RUSBoost Trees (18%). This is consistent with Ahmad's findings that found RUSBoosted performed weakly. Among SVMs, the coarse Gaussian kernel performed best (89% accuracy), followed by medium Gaussian (87%), cubic (84%), and fine Gaussian (82%), matching Bagged Trees (82%) and outperforming fine, medium, and coarse decision tree variants which had accuracy scores of 78%, 61% and 74% respectively. The researchers also evaluated for prediction speed, where RUSBoost and Boosted Trees were able to predict the highest number of observations per second though with poor predictive performance. It should be noted that while ANN's were the third fastest model, making it the best model when looking at speed and accuracy. The decision tree models exhibited faster prediction speed over SVMs, making them a suitable candidate if speed and interpretability are important.

Chen compared a range of machine learning and statistical models using AUC, H-measure, and Brier Score to evaluate classification performance. Results showed that heterogeneous ensemble methods consistently performed best, followed by neural networks, XGBoost, generalized additive models (GAM), and advanced logistic models (GLM2 and GLM3). In contrast, AdaBoost and KNN under-performed, even when compared to the simplest model (GLM1). SVM ranked in the middle of the group. Despite performance declines across all models when using out-of-time testing, the heterogeneous ensembles remained leaders in AUC, while GLM2, GLM3, GAM, and stacked models led in H-measure. Random Forest, AdaBoost, SVM, and KNN consistently lagged behind. This suggests that the complexity of a model is not necessarily associated with its performance and that a simpler statistical models may be suitable for risk classification tasks. 

In their study Krsovytskyi and Stavyskyy compared several classification algorithms, including Logit, LDA, SVM, Random Forest, CART, neural networks, and XGBoost but did not include DTs due to performance issues. To combat data imbalance, the authors tested two techniques: weighting and SMOTE. RUS was not tested. When evaluating with AUROC, SVM and LDA performed the worst on the weighted dataset with a prediction power close to random selection (0.5); their performance improved somewhat after when using SMOTE. Random Forest and XGBoost achieved the best results, with XGBoost outperforming others overall. 

Krsovytskyi and Stavyskyy build off of work by Dokos work on credit risk assessment. Doko’s study compared DTs, Random Forests, SVMs, neural networks, Logistic Regression (LR) models. As the data was imbalanced, Doko tested feature scaling and SMOTE. With feature scaling, Decision Trees performed best, followed by Random Forest, LR, SVM, and neural networks;  without scaling, model performance declined overall. Applying SMOTE unexpectedly worsened performance, likely due to overfitting from synthetic samples. This point is similar to findings by Ahmad where SMOTE introduced noise and did not perform as well as RUS. 

Several points stood out when reviewing these studies to consider while continuing this project. In particular: 
* Accuracy is generally unreliable for imbalanced datasets. Alternative metrics such as F1 score, recall, AUROC, and AUPRC provide a more informative assessment of model performance. In our marketing context, where the cost of a “yes” response may outweigh that of a “no,” recall may remain the preferred metric despite the prevailing preference of F1 score in the studies.
* SMOTE can potentially introduce noise due to its synthetic sample generation. RUS may achieve less noisy outcomes but at the expense of potentially valuable datapoints. Thus SMOTE may still be preferable for smaller datasets or in cases with severe data imbalance, as undersampling to a minority class with few datapoints could weaken our predictors.
* SVM performance is context-dependent. It was the top-performing model for COVID-19 detection (Guhathakurata, 2021) but showed inconsistent results in loan default prediction: among the worst performers in Krsovytskyi and Stavyskyy, mid-range in Chen (2021), and strong but behind ANN and KNN in Azhar et al., indicating that domain or data characteristics may strongly influence outcomes.

### Section 2: SVM Model Training


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

### Conclusion and Recommendations



### References

Ahmad, Amir, Safi, Ourooj, Malebary, Sharaf, Alesawi, Sami, Alkayal, Entisar, Decision Tree Ensembles to Predict Coronavirus Disease 2019 Infection: A Comparative Study, Complexity, 2021, 5550344, 8 pages, 2021. 
 https://onlinelibrary.wiley.com/doi/10.1155/2021/5550344

Azhar Ali, S.E, Rizvi, S. S. H., F. Lai, F., Faizan Ali,  R., and A. Ali Jan Predicting (2021) Delinquency on Mortgage Loans: An Exhaustive Parametric Comparison of Machine Learning Techniques International. Journal of Industrial Engineering and Management, 12 (1), 1. https://www.ijiemjournal.uns.ac.rs/images/journal/volume12/IJIEM_272.pdf

Chen, S., Guo, Z. and X. Zhao (2021). Predicting Mortgage Early Deliquency with Machine Learning Methods. European Journal of Operational Research, 290(1), 358. https://www.sciencedirect.com/science/article/abs/pii/S0377221720306846

Doko, F., Kalajdziski, S., & Mishkovski, I. (2021). Credit risk model based on Central Bank Credit Registry data. Journal of Risk and Financial Management, 14(3), 138. https://www.mdpi.com/1911-8074/14/3/138

Guhathakurata S, Kundu S, Chakraborty A, Banerjee JS. A novel approach to predict COVID-19 using support vector machine. Data Science for COVID-19. 2021:351–64. doi: 10.1016/B978-0-12-824536-1.00014-9. Epub 2021 May 21. [MCID: PMC813796P](https://pmc.ncbi.nlm.nih.gov/articles/PMC8137961/)

Krasovytskyi, D. adn A. Stavyskyy (2024). Predicting Mortgage Loan Defaults Using Machine Learning Techniques. Predicting Mortgage Loan Defaults Using Machine Learning Techniques. Ekonomika, 103(2). https://www.journals.vu.lt/ekonomika/article/download/34809/34141

#### Additional Sources

Bastos, J. A. (2022). Predicting Credit Scores with Boosted Decision Trees. Forecasting, 4(4), 925-935. https://doi.org/10.3390/forecast4040050

Liyanage, C. R., Mago, V., Schiff, R., Ranta, K., Park, A., Lovato-Day, K., Agnor, E., & Gokani, R. (2023). Understanding Why Many People Experiencing Homelessness Reported Migrating to a Small Canadian City: Machine Learning Approach With Augmented Data. JMIR formative research, 7, e43511.  https://pmc.ncbi.nlm.nih.gov/articles/PMC10189624/

Showkat, D., Smith, A. D. R., Wang, L., and A. To. (2023, April). "Who is the right holmess client?": Values in ALgorithmic Homelessness Service Provision and Machine Learning Reasearch  [Conference presentation]. 2023 ACM Conference on Human Factors in Computing Systems (CHI). Hamburg, Germany. https://www.researchgate.net/publication/370131660_Who_is_the_right_homeless_client_Values_in_Algorithmic_Homelessness_Service_Provision_and_Machine_Learning_Research

Tabar, M., Park, H., Winkler, S., Lee. D., Barman-Adhikari, A. and A. Yadav, (2020, August 23-27). Identifying Homeless Youth At-Risk of Substance Use Disorder: Data-Driven Insights for Policymakers [Conference presentation]. KDD'20 Virtual Event. USA. 
https://pike.psu.edu/publications/kdd20-homeless.pdf

Tan, Jialu ()Using machine learning to identify populations at high risk for eviction as an indicator of homelessness [Masters Thesis, Massachusetts Institute of Technology].
https://dspace.mit.edu/handle/1721.1/127660
