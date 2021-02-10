"""
# Author: The code is written by Jawwad Shadman Siddique | R11684947
# Date of Submission:
# The model uses Random Forest Regression Model
# It uses OgalalaNewData1
# Total Raw Data initial = 957
# Total Data after cleaning = 929
# The cleaned dataset is used named as 'ogalanew1_clean,csv'
"""
# Loading Libraries
import os
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import seaborn as sns
from matplotlib import pyplot as plt
import hydroeval as hyd

# Checking Directory

os.getcwd()
os.chdir('D:\Python Practice Programs')
os.getcwd()


# Reading the Dataset
# Using the dataset 'ogallalanewdata1' after cleaning

a = pd.read_csv('ogalanew1_clean.csv') # stratified & cleaned dataset file
X = a.iloc[:,0:31] # Dataframe of the 32 input features
Y = a['log10NO3'] # Series of the output feature - log10(NO3)

# Splitting the dataset into 60% training and 40% testing data

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.4, random_state = 10)

# Performing Grid Search for Best Hyperparameters using 5-fold CV

gsc = GridSearchCV(estimator=RandomForestRegressor(),param_grid={'max_depth': range(3,8),
'n_estimators': (10, 50, 100, 500, 1000)},cv=10, n_jobs=-1)
grid_result = gsc.fit(X_train, Y_train)
best_params = grid_result.best_params_

# Fitting the Random Forest model with train data and predict testing

rfr = RandomForestRegressor(max_depth=best_params["max_depth"],
n_estimators=best_params["n_estimators"],
random_state=False, verbose=False)
rfr.fit(X_train,Y_train)
y_pred = rfr.predict(X_test)
train_pred = rfr.predict(X_train)

# Feature Importance
names = list(X.columns)# Get names of variables
imp = rfr.feature_importances_ # Obtain feature importance
impa = (names,imp) # Make a tuple
impadf = pd.DataFrame(impa) # Write to a dataframe

# Relative Importance Plot
sns.set(style="whitegrid")
ax = sns.barplot(x=imp, y=names) # Make a barplot
ax.set(xlabel="Relative Importance")

# Computing advanced metrics 
# USes Kling Gupta Evaluation Metrics

trainy = np.array(Y_train) # conversion to numpy arrays
testy = np.array(Y_test) # conversion to numpy arrays

kge_train = hyd.kgeprime(train_pred,trainy) # computing metric for train data
kge_test = hyd.kgeprime(y_pred,testy) # computing metric for test data

# User defined function to get the basic metrics of the model 

def metricx(observed,predicted):
    mse = metrics.mean_squared_error(observed,predicted)
    mae = metrics.mean_absolute_error(observed,predicted)
    cor = np.corrcoef(observed,predicted)
    
    zz = [mse, mae, cor]
    return(zz)

# Calling the UDF for metrics calculation

train_metric = metricx(Y_train, train_pred)
test_metric = metricx(Y_test, y_pred)

# printing the metric values

print("The Kling Gupta Metric for Train Data: ", kge_train)
print("The Kling Gupta Metric for Test Data: ", kge_test)
print("The Summary of Metric Accuracy for Train Data: ",train_metric)
print("The Summary of Metric Accuracy for Test Data: ", test_metric)