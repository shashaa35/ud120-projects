#!/usr/bin/python

import pickle
import numpy
numpy.random.seed(42)


### The words (features) and authors (labels), already largely processed.
### These files should have been created from the previous (Lesson 10)
### mini-project.
words_file = "../text_learning/your_word_data.pkl" 
authors_file = "../text_learning/your_email_authors.pkl"
word_data = pickle.load( open(words_file, "rb"))
authors = pickle.load( open(authors_file, "rb") )



### test_size is the percentage of events assigned to the test set (the
### remainder go into training)
### feature matrices changed to dense representations for compatibility with
### classifier functions in versions 0.15.2 and earlier
from sklearn.model_selection import train_test_split
features_train, features_test, labels_train, labels_test = train_test_split(word_data, authors, test_size=0.1, random_state=42)

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.5,
                             stop_words='english')
features_train = vectorizer.fit_transform(features_train)
features_test  = vectorizer.transform(features_test).toarray()
words = vectorizer.get_feature_names()

### a classic way to overfit is to use a small number
### of data points and a large number of features;
### train on only 150 events to put ourselves in this regime
features_train = features_train[:150].toarray()
labels_train   = labels_train[:150]
print((len(features_train)))



### your code goes here

from time import time
from sklearn import tree
from sklearn.metrics import accuracy_score
print((len(features_train[0])))
clf_2 = tree.DecisionTreeClassifier(min_samples_split=40)
t0 = time()

clf_2.fit(features_train,labels_train)
print(("training time:", round(time()-t0, 3), "s"))
# 2.378 s
### fit the classifier on the training features and labels

### use the trained classifier to predict labels for the test features
t1 = time()
pred_2 = clf_2.predict(features_test)
print(("prediction time:", round(time()-t1, 3), "s"))
#  0.267 spred_2 = clf_2.predict(features_test)

# clf_50 = tree.DecisionTreeClassifier(min_samples_split=50)
# clf_50.fit(features_train, labels_train)
# pred_50 = clf_50.predict(features_test)

acc_min_samples_split_2 = accuracy_score(pred_2,labels_test)
# acc_min_samples_split_50 = accuracy_score(pred_50, labels_test)
print("Important features:")
for index, feature in enumerate(clf_2.feature_importances_):
    if feature>0.2:
        print("feature no", index)        
        print("importance", feature)
        print("word", words[index])

print(acc_min_samples_split_2)