#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))
enron_data.pop("TOTAL", 0)
salary = 0.0
minm= 10000000
maxm = 0
for person,data in enron_data.iteritems():
    if(data['salary']!="NaN"):
        maxm = max(maxm,data['salary'])
        minm = min(minm,data['salary'])
        print(str(data['salary']))
print (minm,maxm)
    # salary = salary+1
        # minm = min(minm,int(data['salary']))
        # maxm = max(maxm,int(data['salary']))
    # if data['email_address']!='NaN':
    #   email = email+1
    