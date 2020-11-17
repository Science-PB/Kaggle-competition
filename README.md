# Kaggle-competition
Santander Customer Transaction Prediction


# Introduction
This final report details a variety of classification models used to determine whether a customer from Santander bank would make a transaction, irrespective of the amount of money they transacted. While there were a variety of models that created and tested for this final project, the ultimate goal was to determine the optimal model for this particular problem. 

The data used for this analysis was real customer data provided by Santander bank, but it was anonymized and the features utilized were unknown numeric values. This was done so that Santander bank could use real data while also keeping the privacy of their customers as a priority. There were two files that were provided by Santander: a training dataset and a testing dataset. The training dataset consisted of 200,000 rows, each containing a target variable flag and 200 unknown features. The test dataset only consisted of 30,000 rows and the same 200 unknown features; it did not include a target variable flag, unlike the training data.

The types of classification models used were: K-Nearest Neighbors, Gradient-Boosted Classification Trees, and Logistic Regression. Due to the fact that all three of the aforementioned models were utilized in previous reports over the course of this term, they will not redundantly be explained to the audience. Any further mathematical background or necessary explanations can be found in later sections of the paper as they are deemed necessary for explanation. 


# Methods
The Santander training and test datasets were downloaded and imported into a Jupyter notebook and the necessary packages for all models were loaded. EDA was then performed on both the training and test datasets. There were no null values that needed to be dealt with, and the data was well-formatted. However, because all 200 feature variables were anonymized, there was very little value gained from further EDA. The training data did include target variables and the distribution of the binary values was calculated; the results can be seen in Figure 1 on page 3. 





Due to the nature of this project, and the need to compare models to one another in order to determine the optimal model, the training dataset was used as the primary dataset for the majority of this project. The training data was split randomly into an 80/20 distribution using the train_test_split function. The 80% of the split training data was used to train all three models while the remaining 20% of the training data was used to test the models. Once the optimal model was determined, it was then applied to the provided test data file. For reference, a snapshot of the training dataset can be seen in Figure 2 below.






The first set of models utilized was a set of K-Nearest Neighbors models with n = 1, 5, 11, 21. The KNN package in SciKit Learn was once again used for the creation of these models, and the default parameters were maintained for the models due to the wide variety of models that were created and tested. This meant that all four models utilized the Euclidean Distance rather than the Manhattan Distance and the Uniform Weight rather than the Distance Weight. The Euclidean Distance was maintained for consistency when comparing models; however, the uniform weight was maintained as a way to cut down on some of the processing costs. The code snippet used to create the KNN models can be seen below in Figure 3, while the results of the 4 KNN models can be seen in Table 1. 


