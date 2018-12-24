#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 2 (SVM) mini-project.

    Use a SVM to identify emails from the Enron corpus by their authors:    
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




#########################################################
### your code goes here ###
from sklearn import svm
clf = svm.SVC(kernel='rbf',C=10000)
print str(clf)

# features_train = features_train[:len(features_train)/100] 
# labels_train = labels_train[:len(labels_train)/100] 

t0 = time()
clf.fit(features_train,labels_train)
print "training time:", round(time()-t0, 3), "s"
# 2.378 s
### fit the classifier on the training features and labels

### use the trained classifier to predict labels for the test features
t1 = time()
pred = clf.predict(features_test)
print "prediction time:", round(time()-t1, 3), "s"
#  0.267 s
# print pred[10],pred[26],pred[50]
correct=0
count =0
for ii in range(0,len(pred)):
    if(pred[ii]):
    	count = count+1
    if (pred[ii] == labels_test[ii]):
        correct = correct+1
### calculate and return the accuracy on the test data
### this is slightly different than the example, 
### where we just print the accuracy
### you might need to import an sklearn module
### print clf.score(features_test, labels_test)
accuracy = float(correct)/len(pred)

print accuracy

print count, len(pred)

#########################################################


