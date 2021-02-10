"""
# Author: The code is written by Jawwad Shadman Siddique | R11684947
# Date of Submission: 11 / 25 / 2020
# The model uses MARS / Earth Model
# It uses OgalalaNewData1
# Total Raw Data initial = 957
# Total Data after cleaning = 929
# The cleaned dataset is used named as 'ogalanew1_clean.csv'
"""

#Loading Libraries
import os
import pandas as pd
import numpy as np
from pyearth import Earth
from sklearn import metrics
from sklearn.model_selection import train_test_split
import seaborn as sns
from matplotlib import pyplot as plt
import hydroeval as hyd

# Checking Directory
"""
os.getcwd()
os.chdir('D:\Python Practice Programs')
os.getcwd()
"""

# Reading the Dataset
# Using the dataset 'ogallalanewdata1' after cleaning

a = pd.read_csv('ogalanew1_clean.csv') # stratified & cleaned dataset file
X = a.iloc[:,0:31] # Dataframe of the 32 input features
Y = a['log10NO3'] # Series of the output feature - log10(NO3)

# Splitting the dataset into 60% training and 40% testing data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state = 10)

# Fitting into MARS model

criteria = ('rss', 'gcv')
erth = Earth(feature_importance_type = criteria)
erth.fit(X_train,Y_train)

# Writing out the model summary

print(erth.summary())

# Prediciting the evaluations

train_pred = erth.predict(X_train) 
test_pred = erth.predict(X_test)

# Making the comparison plot

plt.subplot(211)
plt.plot(Y_train, train_pred,'bo')
plt.axline([0,0],[1,1])
plt.xlabel('Observed log $NO_3$')
plt.ylabel('Predicted log $NO_3$')
plt.grid()
plt.title('Model Evaluation - Training')

plt.subplot(212)
plt.plot(Y_test, test_pred, 'ro')
plt.axline([0,0],[1,1])
plt.xlabel('Observed log $NO_3$')
plt.ylabel('Predicted log $NO_3$')
plt.grid()
plt.title('Model Evaluation - Testing')

plt.subplots_adjust(hspace=1)
plt.show()

# User defined function to get the basic metrics of the model 

def metricx(observed,predicted):
    mse = metrics.mean_squared_error(observed,predicted)
    mae = metrics.mean_absolute_error(observed,predicted)
    cor = np.corrcoef(observed,predicted)
    
    zz = [mse, mae, cor]
    return(zz)

# Calling the UDF for metrics calculation

train_metric = metricx(Y_train, train_pred)
test_metric = metricx(Y_test, test_pred)

# Computing advanced metrics 
# USes Kling Gupta Evaluation Metrics

trainy = np.array(Y_train) # conversion to numpy arrays
testy = np.array(Y_test) # conversion to numpy arrays

kge_train = hyd.kgeprime(train_pred,trainy) # computing metric for train data
kge_test = hyd.kgeprime(test_pred,testy) # computing metric for test data

# Variable importance

imp = erth.summary_feature_importances() # extracting the importance values
print(imp) # printing values

# printing the metric values

print("The Kling Gupta Metric for Train Data: ", kge_train)
print("The Kling Gupta Metric for Test Data: ", kge_test)
print("The Summary of Metric Accuracy for Test Data: ",train_metric)
print("The Summary of Metric Accuracy for Test Data: ", test_metric)