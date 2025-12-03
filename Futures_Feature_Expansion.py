#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import pandas as pd
futures = pd.read_csv('processed_futures_monthly_1988_2030.csv')
futures.head()


# In[11]:


futures['implied_rate_diff'] = futures['implied_rate'].diff()
futures.head()


# In[16]:


print(type(futures['exp_date']))


# In[17]:


futures['exp_date'] = pd.to_datetime(df['exp_date']).dt.to_period('M').dt.to_timestamp()
futures.head()


# In[18]:


futures.to_csv("futures_with_diff.csv")


# In[19]:


print(futures.shape)


# In[ ]:




