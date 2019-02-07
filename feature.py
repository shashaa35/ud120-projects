def featureScaling(arr):
    sorted(arr)
    x_min = arr[0]
    x_max = arr[-1]
    part = x_max-x_min
    length = len(arr)
    if(range == 0):
        for i in range(0,length):
            arr[i] = 0.5
    else:
        for i in range(0,length):
            arr[i] = 1.0*(arr[i]-x_min)/part
    return arr

# tests of your feature scaler--line below is input data
data = [115, 140, 175]
print featureScaling(data)