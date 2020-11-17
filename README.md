# Kaggle-competition
Santander Customer Transaction Prediction


# Introduction
This final report details a variety of classification models used to determine whether a customer from Santander bank would make a transaction, irrespective of the amount of money they transacted. While there were a variety of models that created and tested for this final project, the ultimate goal was to determine the optimal model for this particular problem. 

The data used for this analysis was real customer data provided by Santander bank, but it was anonymized and the features utilized were unknown numeric values. This was done so that Santander bank could use real data while also keeping the privacy of their customers as a priority. There were two files that were provided by Santander: a training dataset and a testing dataset. The training dataset consisted of 200,000 rows, each containing a target variable flag and 200 unknown features. The test dataset only consisted of 30,000 rows and the same 200 unknown features; it did not include a target variable flag, unlike the training data.

The types of classification models used were: K-Nearest Neighbors, Gradient-Boosted Classification Trees, and Logistic Regression. Due to the fact that all three of the aforementioned models were utilized in previous reports over the course of this term, they will not redundantly be explained to the audience. Any further mathematical background or necessary explanations can be found in later sections of the paper as they are deemed necessary for explanation. 


# Methods
The Santander training and test datasets were downloaded and imported into a Jupyter notebook and the necessary packages for all models were loaded. EDA was then performed on both the training and test datasets. There were no null values that needed to be dealt with, and the data was well-formatted. However, because all 200 feature variables were anonymized, there was very little value gained from further EDA. The training data did include target variables and the distribution of the binary values was calculated; the results can be seen in Figure 1 on page 3. 


<img width="669" alt="Screen Shot 2020-11-17 at 2 28 07 AM" src="https://user-images.githubusercontent.com/66921930/99359240-c587d000-287c-11eb-843a-ac7730cd2d51.png">


Due to the nature of this project, and the need to compare models to one another in order to determine the optimal model, the training dataset was used as the primary dataset for the majority of this project. The training data was split randomly into an 80/20 distribution using the train_test_split function. The 80% of the split training data was used to train all three models while the remaining 20% of the training data was used to test the models. Once the optimal model was determined, it was then applied to the provided test data file. For reference, a snapshot of the training dataset can be seen in Figure 2 below.


<img width="728" alt="Screen Shot 2020-11-17 at 2 28 48 AM" src="https://user-images.githubusercontent.com/66921930/99359242-c6206680-287c-11eb-8e11-3dc6b4d4cdbd.png">



The first set of models utilized was a set of K-Nearest Neighbors models with n = 1, 5, 11, 21. The KNN package in SciKit Learn was once again used for the creation of these models, and the default parameters were maintained for the models due to the wide variety of models that were created and tested. This meant that all four models utilized the Euclidean Distance rather than the Manhattan Distance and the Uniform Weight rather than the Distance Weight. The Euclidean Distance was maintained for consistency when comparing models; however, the uniform weight was maintained as a way to cut down on some of the processing costs. The code snippet used to create the KNN models can be seen below in Figure 3, while the results of the 4 KNN models can be seen in Table 1. 

<img width="680" alt="Screen Shot 2020-11-17 at 2 29 10 AM" src="https://user-images.githubusercontent.com/66921930/99359248-c6b8fd00-287c-11eb-9d3f-f803f7926b5c.png">

As demonstrated in Table 1 below, the optimal model variant of the KNN models was when n = 11 and n = 21. This made the baseline model accuracy to beat: 90.08%


<img width="407" alt="Screen Shot 2020-11-17 at 2 31 07 AM" src="https://user-images.githubusercontent.com/66921930/99359493-2911fd80-287d-11eb-9fae-23b2d3a03b34.png">


Next, the gradient boosting model was created using the same SciKit learn package and methodology used for the third assignment of this class. The code snippet detailing the methods of Gradient Boosting can be seen in Figure 4 below.



<img width="621" alt="Screen Shot 2020-11-17 at 2 31 28 AM" src="https://user-images.githubusercontent.com/66921930/99359494-2911fd80-287d-11eb-9bbb-ebeb060a84ac.png">


As demonstrated by Figure 4 on page 4, the output of the gradient boosting model was 90.43% accuracy, which was higher than the KNN accuracy of 90.08%, thus making gradient boosting the new standard model to beat. Only one gradient boosting model was made because of some processing limitations, and the parameters were intentionally kept as the defaults from the SciKit learn package. The learning rate was set to 0.1, which allowed for more minute changes and less of a potential for creating an over-fit model. The number of estimators (N_estimators) was kept at 100, because it again allowed for the creation of a robust, but not over-fit model. Lastly, the max depth of the trees was set to 3, thus allowing for faster performance when creating the model. 

In addition to the creation of the gradient boosting model, a 5-fold cross-validation test was performed in order to ensure the accuracy of the model. The 5-fold validation was 90.37%, which was still higher than the KNN model and was only 0.06% off of the gradient boosting model accuracy score. 

The final model created was a simple logistic regression model for a simple binary classification problem. Before the model was created, the GridSearchCV function was used to determine the optimal parameters of the model. The code snippet of this method can be seen in Figure 5 below. 


<img width="534" alt="Screen Shot 2020-11-17 at 2 31 52 AM" src="https://user-images.githubusercontent.com/66921930/99359496-29aa9400-287d-11eb-8763-9c92fc1065ec.png">


The optimal parameters for this particular logistic regression problem were a C value of about 19307, indicating that there was very low regularization and an L2 penalty. The L2 penalty is also known as the Ridge Regression and its loss function is simply a squared magnitude of the coefficient. L1, or Lasso, Regression alternatively adds the absolute value of magnitude of the coefficient as the penalty term in the loss function. This ultimately eliminates some features, which is extraordinarily beneficial for datasets with a very large number of features. The equation for the L1 loss function can be seen in Figure 6 on page 6. 


<img width="505" alt="Screen Shot 2020-11-17 at 2 34 02 AM" src="https://user-images.githubusercontent.com/66921930/99359889-bd7c6000-287d-11eb-8a97-0ac3eaa46d1a.png">


The logistic regression model was then created with the established parameters, and subsequently run and tested with the Santander training data. The code snippet outlining this process can be seen below in Figure 7. 


<img width="752" alt="Screen Shot 2020-11-17 at 2 34 27 AM" src="https://user-images.githubusercontent.com/66921930/99359890-be14f680-287d-11eb-8929-b2c5ac4466f2.png">



As demonstrate in Figure 7, the logistic regression model had a 0.9110 accuracy when comparing to the known transaction outcomes. To ensure model accuracy, another 5-fold cross validation was utilized on the logistic regression model, which gave an output of 91.04% accuracy. This was still higher than the KNN and gradient boosting accuracies. The final 91.10% accuracy was the highest of all models and will therefore be the final model used for the test dataset.
Results and Discussion
While logistic regression ultimately output the best model based on the accuracy score, it is still important to note that all the models were within a few percentage points of each other, and that some continued parameter adjustments might alter which model ultimately gave the highest accuracy. As it stands, KNN was the least effective model in predicting whether or not a customer would make a transaction. Figure 8 is a confusion matrix of the results from the KNN = 21 model. When looking at the bottom right-hand quadrant, one can see that there were not accurate predictions made for instances where a customer did make a transaction. Because the training data set did have so few known transactions, it is not surprising that this model did not catch any of them. 



<img width="552" alt="Screen Shot 2020-11-17 at 2 35 12 AM" src="https://user-images.githubusercontent.com/66921930/99359893-bead8d00-287d-11eb-917f-ec96178345c6.png">
<img width="529" alt="Screen Shot 2020-11-17 at 2 35 18 AM" src="https://user-images.githubusercontent.com/66921930/99359894-bead8d00-287d-11eb-88bf-e2836afed544.png">



The gradient boosting model (Figure 9, page 7) did show improvement when predicting the transaction incidences, with an increased value of 28 correct predictions. While the KNN model was not too far off from the prediction accuracy of gradient boosting, boasting a false positive rate of 0.00%, it also missed every single positive transaction, thus bringing into question the efficacy of the model. 
While gradient boosting did see an increase in the false positive rate, up to 0.13%, it also saw an increase in its ability to predict true negatives at 4.71%, which was a significant increase when compared to KNN



<img width="604" alt="Screen Shot 2020-11-17 at 2 36 09 AM" src="https://user-images.githubusercontent.com/66921930/99359897-bead8d00-287d-11eb-98f9-b78d785f46e1.png">


Figure 10 shows the confusion matrix for the logistic regression model. Again, there was an increase in the false positive rate, this time it was over one percent at 1.35%. However, it did boast the highest overall accuracy overall at 91.10%, which is the highest of all the models. Because of its continuously low false positive rate and the fact that it has the highest overall accuracy, logistic regression can be considered the optimal model for this particular solution. 
Once decided, the logistic regression model was applied to the original Santander test data set of 30,000 rows, and it found that there were 0 instances of transactions. No final accuracy could be determined, but when compared with other results from the Kaggle competition, it was findings were similar to the contest winners, which was intriguing anecdotal confirmation of the model’s ability to predict transactions. 

# Conclusion
It is not always easy to determine the best model for a given problem and dataset when first presented with a project. While things like a distribution and the type of predictions being made can immediately eliminate a good proportion of models, it is not always straightforward to find the best one of the remaining models. Often times, it takes multiple iterations of multiple model types in order to determine the best model. 

Even with all the interesting machine learning models that are continuously being developed and re-developed, a classic logistic regression for a binary classification problem is still often times the most robust and accurate option. 

If this project was to be continued, and more processing power was available, the gradient boosting and KNN models would have been tuned a bit more, which could have made them more accurate than the logistic regression model. As it stands though, logistic regression was the most accurate and was ultimately deemed the optimal model.


# References
Agresti, A. (2015). Foundations of linear and generalized linear models. ProQuest Ebook Central https://ebookcentral.proquest.com

Maklin, Cory. (July 2019). K Nearest Neighbor Algorithm In Python. Towards Data Science. Retrieved from https://towardsdatascience.com/k-nearest-neighbor-python-2fccc47d2a55 

Santander Customer Transaction Prediction. (Feb 2019). Kaggle [Data set]. Retrieved from https://www.kaggle.com/c/santander-customer-transaction-prediction/overview 

Sklearn.ensemble.GradientBoostingClassifier. (n.d.). SciKit Learn. Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html#sklearn.ensemble.GradientBoostingClassifier 

Sklearn.linear_model.LogisticRegression. (n.d.). SciKit Learn. Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html#sklearn.linear_model.LogisticRegression.predict_log_proba 

Sklearn.neighbors.KNeighborsClassifier. (n.d.). SciKit Learn. Retrieved from https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html#sklearn.neighbors.KNeighborsClassifier 

StatQuest with Josh Starmer. (8 April 2019). Gradient Boost Part 3: Classification [Video]. YouTube. https://www.youtube.com/watch?v=jxuNLH5dXCs 

StatQuest with Josh Starmer. (22 April 2019). Gradient Boost Part 4: Classification Details [Video]. YouTube. https://www.youtube.com/watch?v=StWY5QWMXCw 

Vanderplas, Jake. (n.d.). “1.6 Nearest Neighbors.” SciKit Learn. Retrieved from https://scikit-learn.org/stable/modules/neighbors.html 

1.11 Ensemble Methods. (2020). SciKit Learn. Retrieved from https://scikit-learn.org/stable/modules/ensemble.html#gradient-tree-boosting 
