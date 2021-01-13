#!/usr/bin/env python
# coding: utf-8

# In[45]:


import tensorflow as tf
import numpy as np
import pandas as pd

from sklearn import preprocessing 


# In[46]:


### loading of the data
raw_data = pd.read_csv("heart.csv")
all_inputs = raw_data.drop(["target"], axis =1)
all_targets = raw_data["target"]


# In[47]:


### preprossing the data to be properly be used within the network code 
scaled_inputs = preprocessing.scale(all_inputs)

sample_count = all_inputs.shape[0]

numTraining_samples = int(0.8*sample_count)
numValidation_samples = int(0.1*sample_count)
numTest_samples = sample_count - (numTraining_samples+numValidation_samples)

train_inputs=scaled_inputs[:numTraining_samples]
train_targets=all_targets[:numTraining_samples]

A =numTraining_samples+numValidation_samples

validation_inputs= scaled_inputs[numTraining_samples:A]
validation_targets= all_targets[numTraining_samples:A]

test_inputs =scaled_inputs[A:]
test_targets =all_targets[A:]


# In[48]:


### only two outputs 0 or 1
outputSize = 2
HiddenLayerSize = 70

### the network code itself is listed below using relu for the first and second layers and a sigmoid function for the final in order to better assume the 0 or 1 output
model = tf.keras.Sequential([
    tf.keras.layers.Dense(HiddenLayerSize, activation = "relu"),
    tf.keras.layers.Dense(HiddenLayerSize, activation = "relu"),
    ### now for the output layer
    tf.keras.layers.Dense(outputSize, activation="sigmoid")
])
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])


# In[49]:


batch_size = 100
epochs = 80

### will stop the function early if the model begins to decrease to prevent overfitting to the training data
### patience is marked by the number of times the model will allow for this to occur. In this case its allowed twice
early_stopping = tf.keras.callbacks.EarlyStopping(patience = 2)

model.fit(train_inputs, 
          train_targets,
         batch_size=batch_size, 
         epochs=epochs,
         callbacks=[early_stopping],
         validation_data = (validation_inputs, validation_targets),
         verbose =2 )


# In[23]:


test_loss, test_accuracy = model.evaluate(test_inputs, test_targets)
print("test loss:", test_loss, "test accuracy:", test_accuracy)


# In[ ]:


### on average a 97% accuracy of the program

