# TitanicDataset
Different machine learning approaches for the Kaggle Titanic Dataset

## Feature Engineering

The 'Name', 'Ticket', 'Cabin', and 'Embarked' fields were dropped as I judged them to have no relevance to the survival rate. 

## Data Cleaning

The 'Sex' column was replaced by the one-hot encoded columns of male and female. The missing gaps in 'Age' and 'Fare' columns were replaced by their mean values. 

## Random Forest Classification

Using sklearn, a random forest classifier was used on the dataset. 

The training dataset from train.csv was further subdivided into a 75-25 training/validation split.

Grid search 3-fold cross validation was done on the n_estimators and max_depth hyperparameters, giving a best fit at n_estimators = 64 and max_depth = 8 for the Random Forest Classifier.

An accuracy of 76.794% was achieved on the dataset of test.csv, when the model was fit on the 75% training split from train.csv. After obtaining the best fit hyperparameters from the grid search, the model was trained on the entire training dataset and achieved a final accuracy of 77.511% on the test.csv data. 

UPDATE: Further grid search analysis on tighter, closer ranges was done to find better hyperparameters at n_estimators = 39 and max_depth = 11. However, this only produced an accuracy of 76.555%, the worst one yet. This may indicate either overfitting or the dataset being too small in capacity. I would advise against this much heavy optimizing. 

## Gradient Boosted Trees Classification

A gradient boosted tree classification was trained on this dataset using the sklearn module.

Grid search 3-fold cross validation was done on the n_estimators=200, max_depth=4 and learning_rate=0.01 parameters. 

Using the best estimators, an accuracy of 76.555% was achieved, which was also seen before on the overfit random forest classification. This model does not turn out to be better.

## Voting Classification

A voting classifier from the sklearn package was used for the purposes of this classification.

A voting classification was done on the basis of hard majority rule voting, from classifiers: Linear SVC, Random Forest, Gradient Boosted Trees, and Logistic Regression.

Grid search 3-fold cross validation was done to find the optimal settings of each classifier. Random Forest and Gradient Boosted Trees optimal settings have already been discussed as separate classification attempts. SVC was found optimal for the linear kernel, and logistic regression classifier had essentially the same accuracy and output regardless of regularization settings.

This ensemble of classifiers produced a much higher accuracy of 78.468% compared to earlier attempts. 
