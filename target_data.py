#!/usr/bin/env python
# coding: utf-8

# # Fed interest rate decision labels

# Notes:
# - Fed switches from single target interest rate to a target interest rate range in 2008
# - We use the target interest rate pre 2008, and the upper limit of the target interest rate range post 2008

# Data comes in as the actual rate. We need to parse it by rate decision.

# In[1]:


import pandas as pd


# In[2]:


rates_pre_2008 = pd.read_csv('DFEDTAR.csv')
rates_post_2008 = pd.read_csv('DFEDTARU.csv')


# In[3]:


rates_pre_2008.head()


# In[4]:


rates_pre_2008 = rates_pre_2008.rename(columns={'DFEDTAR':'rate'})


# In[5]:


rates_post_2008.head()


# In[6]:


rates_post_2008 = rates_post_2008.rename(columns={'DFEDTARU':'rate'})


# In[8]:


full_rates = pd.concat([rates_pre_2008, rates_post_2008], axis=0, ignore_index=True)


# In[9]:


full_rates.head()


# In[10]:


full_rates['observation_date'] = pd.to_datetime(full_rates['observation_date'])


# In[11]:


full_rates[:10]


# # Re-read beige book data

# In[12]:


beige_book_data = pd.read_csv('beige_book_1996_2025.csv')


# In[13]:


beige_book_data.head()


# In[14]:


filtered = full_rates[full_rates['observation_date'].dt.day == 1].copy()


# In[15]:


filtered.head()


# # Trim table based on beige book dates

# In[16]:


year_month_pairs = [(year, month) for year, month in zip(beige_book_data['year'], beige_book_data['month'])]


# In[17]:


print(year_month_pairs)


# In[18]:


type(full_rates['observation_date'][0].month)


# In[19]:


rates_filtered = filtered[
    (full_rates['observation_date'].dt.to_period('M').apply(lambda x: (x.year, x.month)).isin(year_month_pairs))
]


# In[26]:


rates_filtered = rates_filtered.copy()


# In[27]:


rates_filtered.head()


# In[28]:


decisions = []
for prev, curr in zip(rates_filtered['rate'][:-1], rates_filtered['rate'][1:]):
    if curr > prev:
        decisions.append('raise')
    elif curr < prev:
        decisions.append('lower')
    else:
        decisions.append('hold')
decisions.insert(0, "na")


# In[29]:


len(decisions)


# In[30]:


rates_filtered['decision'] = decisions


# In[31]:


rates_filtered.to_csv("filtered_labels.csv")

