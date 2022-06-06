# ML project 

My first repository on GitHub!
This is a machine learning algorithm implementation on the Epilepsy dataset acquired from the following Kaggle link:
[Epileptic seizures dataset | Kaggle](https://www.kaggle.com/datasets/chaditya95/epileptic-seizures-dataset)


Seizures associated with abnormal brain activities caused by epileptic disorder is widely typical and has many symptoms, such as loss of awareness and unusual behavior as well as confusion. 
In this project, a classification of the Epileptic Seizure dataset was done using various classification algorithms – KNN, SVM, Random Forest Classifier and Decision Tree CLassifier. After implementation of the algorithm, the genetic algorithm has been used to optimize the KNN classification model. 
The aim of this project is to determine the most suitable classification algorithm to classify the epileptic seizure dataset to determine whether a person would have a seizure in the given circumstances.


Only samples associated with class 1 had an Epileptic Seizure, and therefore, our analysis will take a binary shape for Epileptic Seizure and non- Epileptic Seizure cases, which contain classes {2,3,4,5}. So, data preprocessing involves binarizing the data into classes 0 and 1.
Additionally, there are 178 columns in the original dataset so preprocessing stage also involves selection of a few selective columns which could explain most of the data to disallow unnecessary features.


Training models involved:


1. KNN classifier: The best value of K is chosen from 1-20 based on error rate analysis. Accordingly, 3 is chosen as the value of K and the model is built accordingly. Further, cross validation is performed over the dataset giving good accuracy scores.
3. SVM classifier: Using hyperparameter optimization, the best possible parameter values seem to be the ones that have been used in the code snippet.
4. Random Forest Classifier: Creates decision trees based on entropy split and understands the best split accuracy.
5. Decision tree classifier: Builds a single decision tree based on given parameters. This would be an unlikely choice since the decision tree has many splits and also results in a not so good accuracy.
