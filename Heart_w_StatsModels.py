#!/usr/bin/env python
# coding: utf-8

# In[9]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import seaborn as sns
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn import preprocessing


# In[10]:


### the following dataset was taken from 1988 study from  Cleveland, Hungary, Switzerland, and Long Beach
### the model was proposed to be 
raw_data = pd.read_csv("heart.csv")


# In[11]:


### spliting the data
a_train, a_test = train_test_split(raw_data, test_size = 0.2)

### transforming the dataset
scaler = preprocessing.StandardScaler().fit(a_train)
train_scaled = scaler.transform(a_train)

scaler = preprocessing.StandardScaler().fit(a_test)
test_scaled = scaler.transform(a_test)

### removal of negligent p-values after result.summary() is taken
targets = a_train["target"]
inputdata = a_train.drop(["target"], axis =1)
inputdata = inputdata.drop(["age"], axis =1)
inputdata= inputdata.drop(["fbs"], axis= 1)

test_actual = a_test["target"]
test = a_test.drop(["target"], axis =1)
test = test.drop(["age"], axis =1)
test= test.drop(["fbs"], axis= 1)


# In[5]:


### establishment of a constant and the building of a logit function based of the presented data
x0 = sm.add_constant(inputdata)
reg_log = sm.Logit(targets, x0)
result = reg_log.fit()


# In[6]:


result.summary()


# In[7]:


### now for setting up a confusion matrix
#### can be done with the following np function below
np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})
result.pred_table()


# In[8]:


### or it can be done via the following with a pandas dataframe
confusMatrix = pd.DataFrame(result.pred_table())
confusMatrix.columns = ["predict 0", "predict 1"]
confusMatrix = confusMatrix.rename(index={0: "Actual 0", 1: "Actual 1"})
print(confusMatrix)


# In[9]:


CM = np.array(result.pred_table())
accuracy_train = ((CM[0,0] + CM[1,1])/CM.sum())
print(accuracy_train)


# In[10]:


test_constant = sm.add_constant(test)
test


# In[13]:


# 85% accuracy 

