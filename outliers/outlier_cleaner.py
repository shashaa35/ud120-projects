#!/usr/bin/python


def outlierCleaner(predictions, ages, net_worths):
    """
        Clean away the 10% of points that have the largest
        residual errors (difference between the prediction
        and the actual net worth).

        Return a list of tuples named cleaned_data where 
        each tuple is of the form (age, net_worth, error).
    """
    
    cleaned_data = []

    ### your code goes here
    for i in range(0, len(predictions)):
        cleaned_data.append(tuple((ages[i],net_worths[i],abs(predictions[i]-net_worths[i]))))

    # print (str(cleaned_data))
    cleaned_data = sorted(cleaned_data, key=lambda x: x[2])    
    # print (str(cleaned_data))
    return cleaned_data[0:int(0.9*len(cleaned_data))]

