#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 3 (decision tree) mini-project.

    Use a Decision Tree to identify emails from the Enron corpus by author:    
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()

from sklearn import tree
from sklearn.metrics import accuracy_score
print(len(features_train[0]))
clf_2 = tree.DecisionTreeClassifier(min_samples_split=40)
t0 = time()

clf_2.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"
# 2.378 s
### fit the classifier on the training features and labels

### use the trained classifier to predict labels for the test features
t1 = time()
pred_2 = clf_2.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"
#  0.267 spred_2 = clf_2.predict(features_test)

# clf_50 = tree.DecisionTreeClassifier(min_samples_split=50)
# clf_50.fit(features_train, labels_train)
# pred_50 = clf_50.predict(features_test)

acc_min_samples_split_2 = accuracy_score(pred_2,labels_test)
# acc_min_samples_split_50 = accuracy_score(pred_50, labels_test)

print (acc_min_samples_split_2)

#########################################################
### your code goes here ###


#########################################################


