#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

from sklearn import preprocessing
sns.set()


# In[3]:


raw_data = pd.read_csv("heart.csv")

train, test = train_test_split(raw_data, test_size = 0.2)
train_targets = train["target"]
train = train.drop(["target"], axis =1)

scaler = preprocessing.StandardScaler().fit(train)
train_scaled = scaler.transform(train)


# In[5]:


log_reg = LogisticRegression().fit(train_scaled, train_targets)


# In[6]:


### shaping test targets to be used for the test of the model
test_targets = test["target"]
test= test.drop(["target"], axis=1)
test_scaled = scaler.transform(test)
final = log_reg.predict(test_scaled)


# In[1]:


### function for a confusion matrix to better define the accuracy of the model
def ConfusionMatrix(data, actual_values, model):
    predictedvalues = model.predict(data)
    #### only part im confused about is this bins section
    bins = np.array([0, 0.5, 1])
    confusionM = np.histogram2d(actual_values, predictedvalues, bins=bins)[0]
    accuracy = ((confusionM[0,0]+confusionM[1,1])/confusionM.sum())
    return confusionM, accuracy


# In[8]:


accuracy_array = np.array(final == test_targets)

posistions = len(accuracy_array)
counting_scaler = 0
for i in range(posistions):
    it = accuracy_array[i]
    if it:
        counting_scaler += 1

accuracy = str((counting_scaler/posistions)*100)+"%"
print(accuracy)


# In[9]:


### as high as 87 percent accuracy with sklearn. Thats two percent better than statsmodels 
### differences within this output may be caused by the manner in which the train_test_split is arranged


# In[ ]:




