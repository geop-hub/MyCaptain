#!/usr/bin/env python
# coding: utf-8

# In[9]:


#importing dependcies
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[2]:


data = pd.read_csv('mnist_test.csv')


# In[4]:


data.head()


# In[14]:


a=data.iloc[2,1:].values
a=a.reshape(28,28).astype('uint8')
plt.imshow(a)


# In[6]:


data_x=data.iloc[:,1:]
data_y=data.iloc[:,0]


# In[7]:


X_train,X_test,y_train,y_test=train_test_split(data_x,data_y,random_state=1,test_size=0.2)


# In[20]:


rfc=RandomForestClassifier(n_estimators=10)


# In[21]:


rfc.fit(X_train,y_train)


# In[22]:


pred=rfc.predict(X_test)


# In[23]:


rfc.score(X_test,y_test)


# In[ ]:




