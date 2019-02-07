#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)



### your code goes here 



### it's all yours from here forward!  
from sklearn.model_selection import train_test_split
from time import time
features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.3, random_state=42)

t0 = time()

from sklearn import tree
from sklearn.metrics import accuracy_score
clf = tree.DecisionTreeClassifier()
clf.fit(features_train, labels_train)

print("training time:"+ str(round(time()-t0, 3)) + "s")

t1 = time()
pred = clf.predict(features_test)
print("prediction time:"+ str(round(time()-t1, 3))+ "s")

print("Accuracy score is: " + str(accuracy_score(pred,labels_test)))
# print(labels_test.count(1))
# print(len(labels_test))
# for x in xrange(1,len(pred)):
# 	print labels_test[x],pred[x]
# print(labels_test)
# print(pred)
from sklearn.metrics import precision_score,recall_score
print(precision_score(labels_test,pred))
print(recall_score(labels_test,pred))
