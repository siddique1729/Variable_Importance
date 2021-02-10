"""
# The code is written by Jawwad Shadman Siddique | R11684947
# Date of Submission: 11 / 24 / 2020
# This Step performs Model Entropy & Correlation
# It uses OgalalaNewData1
# Total Raw Data initial = 957
# Total Data after cleaning = 929
# The cleaned dataset is used named as 'ogalanew1_clean,csv'
"""

# importing required libraries & packages

import os
import pandas as pd 
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.feature_selection import mutual_info_regression
from statsmodels.stats.outliers_influence import variance_inflation_factor 

# Checking Directory

os.getcwd()
os.chdir('D:\Python Practice Programs')
os.getcwd()

a = pd.read_csv('ogalanew1_clean.csv')


# correlation calculation with all the data points 
# store the calculated correlation values into the corr variable

corr = a.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))

f, ax = plt.subplots(figsize=(15,15))
cmap = sns.diverging_palette(220, 10, as_cmap=True)
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=0.3, center=0, square=True, 
            linewidths=0.5, cbar_kws={"shrink": 0.5})

# Calculation of Mutual Information Criteria

Y = a['log10NO3']
X = a.iloc[:,0:31] 
# Need to clean data with non-standard value 'U' 
# Otherwise will show error when calculating from 0 to 33 (total column values)


# Calculating Model Entropy

MI = mutual_info_regression(X,Y)
MI = MI*100/np.max(MI)


# Plot Showing important variables based on Mutual Information

cols = list(a.columns)[0:31]
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.bar(cols,MI)
plt.ylabel('Rel. Mutual Information')
plt.xticks(rotation='vertical')
plt.grid(True)
plt.show()
#plt.savefig("04_Mutual Information.png",bbox_inches='tight')