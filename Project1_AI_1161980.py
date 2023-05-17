#!/usr/bin/env python
# coding: utf-8

# In[1]:


#importing Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
import pickle
import seaborn as sns
import tensorflow as tf


# In[2]:


#importing the data
data_file = pd.read_csv("C:\\Users\\dell\\Desktop\\ADANIENTNS.csv")


# In[3]:


#Checking the data
data_file.head()


# In[4]:


#understanding the data
data_file.describe()


# In[5]:


data_file.columns


# In[6]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig , ax = plt.subplots(figsize=(20,20))
sns.boxplot(data=data_file, ax=ax)


# there are so many outlieres in volume, will remove them


# In[7]:


new_data_file = data_file[data_file['Volume'] < data_file['Volume'].quantile(q=0.75)]


# In[8]:


get_ipython().run_line_magic('matplotlib', 'inline')
fig , ax = plt.subplots(figsize=(20,20))
sns.boxplot(data=new_data_file, ax=ax)


# In[9]:


new_data_file['Date'] = pd.to_datetime(new_data_file['Date'])


# In[10]:


new_data_file.set_index('Date',inplace = True)


# In[11]:


#dividing the training and test data
X_data = new_data_file.drop('Close', axis=1)
Y_data = new_data_file['Close']


# In[12]:


X_data.describe()


# In[13]:


Y_data.describe()


# In[14]:


# We can observe that data is not standerdised, so transforming the data for training and test data

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_data)


# In[15]:


fig,ax = plt.subplots(figsize=(20,20))
sns.boxplot(data=X_scaled,ax=ax)


# In[16]:


X_train ,X_test, y_train, y_test = train_test_split(X_scaled,Y_data,test_size=20, random_state = 0)


# In[17]:


# Creating the model with ADAM optimizer
model = tf.keras.Sequential([
    tf.keras.layers.Dense(64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')


# In[18]:


# Train the model on the training data
model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test))


# In[19]:


from sklearn.linear_model import LinearRegression
logi = LinearRegression()
logi.fit(X_train,y_train)
pred = logi.predict(X_test)


# In[20]:


mean_absolute_error = tf.keras.losses.mean_absolute_error(y_test, pred)
mean_absolute_error = mean_absolute_error.numpy().mean().item()
print(f'your mean absolute error is :{mean_absolute_error:.2f}')


# In[21]:


# Plot the predicted and actual values
plt.plot(y_test.index, pred, label='Predicted'),
plt.plot(y_test.index, y_test, label='Actual'),
plt.xlabel('Time'),
plt.legend(),
plt.show()

