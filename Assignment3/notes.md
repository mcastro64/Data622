

Krsovytskyi - Compared Logit, LDA, SVM, RF, CART, NNET, XG8TREE. SVM and LDA showed worst perfomance (very similar to random guess) on imbalanced but weighted data (did not predict any defaults), did a little better after SMOTE. Logit overpredicted defaults (real 564, predicted 3000+)/ RF and XGBTree did the best. Most models liked GDP as highest IV, but not XGBTree(RF did not when weighting but did when SMOTE). XGBTree liked DSTI. No simple DT.

Doko  -  DT > RF > LR > SVM > NN for imbalanced data with scaling. did surprisingly worst with SMOTE with balancing (so not considered) and not as good without scaling. Perhaps this has to do with overfitting? Delayed Days best IV. Age not important.

Azhar et al - 19 ML tested including upport
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

Azhar Ali, S.E, Rizvi, S. S. H., F. Lai, F., Faizan Ali,  R., and A. Ali Jan Predicting  (2021) Delinquency on Mortgage Loans: An Exhaustive Parametric Comparison of Machine Learning Techniques International. Journal of Industrial Engineering and Management, 12 (1), 1. https://www.ijiemjournal.uns.ac.rs/images/journal/volume12/IJIEM_272.pdf

Bastos, J. A. (2022). Predicting Credit Scores with Boosted Decision Trees. Forecasting, 4(4), 925-935. https://doi.org/10.3390/forecast4040050

Chen, S., Guo, Z. and X. Zhao (2021). Predicting Mortgage Early Deliquency with Machine Learning Methods. European Journal of Operational Research, 290(1), 358. https://www.sciencedirect.com/science/article/abs/pii/S0377221720306846

Doko, F., Kalajdziski, S., & Mishkovski, I. (2021). Credit risk model based on Central Bank Credit Registry data. Journal of Risk and Financial Management, 14(3), 138. https://www.mdpi.com/1911-8074/14/3/138

Krasovytskyi, D. adn A. Stavyskyy (2024). Predicting Mortgage Loan Defaults Using Machine Learning Techniques. Predicting Mortgage Loan Defaults Using Machine Learning Techniques. Ekonomika, 103(2). https://www.journals.vu.lt/ekonomika/article/download/34809/34141

#### Additional Sources

Tabar, M., Park, H., Winkler, S., Lee. D., Barman-Adhikari, A. and A. Yadav, (2020, August 23-27). Identifying Homeless Youth At-Risk of Substance Use Disorder: Data-Driven Insights for Policymakers [Conference presentation]. KDD'20 Virtual Event. USA. 
https://pike.psu.edu/publications/kdd20-homeless.pdf

Tan, Jialu (2020)Using machine learning to identify populations at high risk for eviction as an indicator of homelessness [Masters Thesis, Massachusetts Institute of Technology].
https://dspace.mit.edu/handle/1721.1/127660

Liyanage, C. R., Mago, V., Schiff, R., Ranta, K., Park, A., Lovato-Day, K., Agnor, E., & Gokani, R. (2023). Understanding Why Many People Experiencing Homelessness Reported Migrating to a Small Canadian City: Machine Learning Approach With Augmented Data. JMIR formative research, 7, e43511.  https://pmc.ncbi.nlm.nih.gov/articles/PMC10189624/

Showkat, D., Smith, A. D. R., Wang, L., and A. To. (2023, April). "Who is the right holmess client?": Values in ALgorithmic Homelessness Service Provision and Machine Learning Reasearch  [Conference presentation]. 2023 ACM Conference on Human Factors in Computing Systems (CHI). Hamburg, Germany. https://www.researchgate.net/publication/370131660_Who_is_the_right_homeless_client_Values_in_Algorithmic_Homelessness_Service_Provision_and_Machine_Learning_Research