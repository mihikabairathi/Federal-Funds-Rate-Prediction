#!/usr/bin/env python
# coding: utf-8

# # Loughran-McDonald Dictionary for Sentiment Analysis

# Data reference: [Notre Dame](https://sraf.nd.edu/loughranmcdonald-master-dictionary/)
# 
# Processing reference: [Wharton](https://wrds-www.wharton.upenn.edu/pages/classroom/sec-filings-dictionary-based-sentiment-analysis/)

# ## Load in dictionary

# In[1]:


import csv
import glob
import re
import string
import sys
import datetime as dt


# In[2]:


def utf8len(s):
    """helper function to get the size of string"""
    return len(s.encode("utf-8"))


# In[3]:


# Load your master dictionary file. This file requires a
# Word column and a Syllables column. Other columns are optional
# and should be defined in the SENTIMENT_OUTPUT_FIELDS Python dictionary below.
master_dictionary_file = "Loughran-McDonald_MasterDictionary_1993-2024.csv"


# In[4]:


# Load the master dictionary CSV file into a Python dictionary
# with Word as the key.
master_dictionary = {}
with open(master_dictionary_file) as csv_file:
    csv_reader = csv.DictReader(csv_file, delimiter=",")
    line_count = 0
    for row in csv_reader:
        master_dictionary[row["Word"].lower()] = row
        line_count += 1
print(f"master dictionary has {len(master_dictionary)} words.")


# The dictionary is now loaded into memory. Let's inspect what information it contains for an example word.

# In[5]:


master_dictionary["key"]


# Normalize words by lowercasing them all

# In[6]:


for key, item in master_dictionary.items():
    item = {item_key.lower(): item_v for item_key, item_v in item.items()}
    item['word'] = item['word'].lower()
    master_dictionary[key] = item
    
    
print(f"master dictionary has {len(master_dictionary)} words.")


# Convert numeric fields to numeric

# In[7]:


for key, item in master_dictionary.items():
    for field, value in item.items():
        if type(value) == str and value.isdigit():
            item[field] = int(value)
        elif type(value) == str:
            try:
                item[field] = float(value)
            except:
                pass
    master_dictionary[key] = item


# In[8]:


master_dictionary["key"]


# In[9]:


type(master_dictionary['key']['word proportion'])


# Sentiment scores are given *BY YEAR* added, but actual values are *categorical*.

# ## Calculate score of document based on sentiment
# 
# - Assumes input of list of words/tokens
# - Assumes we are looking for negative, positive, and uncertainty
# - The following sentiments will be excluded: litigious, strong_modal, weak_modal, and constraining

# In[10]:


# The SENTIMENT_OUTPUT_FIELDS list below contains the sentiment fields we want
# to include.
SENTIMENT_OUTPUT_FIELDS = [
    "negative",
    "positive",
    "uncertainty",
]


# In[11]:


# Assumes doc has been cleaned and lowercased
def calculate_sentiment_score(doc: list[str]):
    token_count = 0
    sentiment_counts = {k: 0 for k in SENTIMENT_OUTPUT_FIELDS}
    for token in doc:
        if token in master_dictionary:
            token_count += 1
            for sentiment in SENTIMENT_OUTPUT_FIELDS:
                sentiment_counts[sentiment] += int(master_dictionary[token][sentiment] != 0)
    return {k: v / token_count for k, v in sentiment_counts.items()}
                


# In[12]:


test_doc = "terrible horrible very bad day".split(" ")


# In[13]:


for token in test_doc:
    print(master_dictionary[token])


# In[14]:


calculate_sentiment_score(test_doc)


# In[15]:


test_doc2 = "happy sunny awesome ice cream sundae cool".split(" ")


# In[16]:


for token in test_doc2:
    print(master_dictionary[token])


# In[17]:


calculate_sentiment_score(test_doc2)


# ## Now use python module LMSentimentDict

# In[18]:


from importlib import reload
from lm_sentiment import LMSentimentDict


# In[19]:


sentiment_dict = LMSentimentDict(master_dictionary_file, SENTIMENT_OUTPUT_FIELDS)


# In[20]:


sentiment_dict.master_dictionary['happy']


# In[25]:


print(test_doc, test_doc2)


# In[21]:


print(sentiment_dict.calculate_sentiment_score(test_doc))
print(sentiment_dict.calculate_sentiment_score(test_doc2))


# ## Run on begie_book_1996_2025.csv

# In[22]:


import pandas as pd


# In[36]:


bbdf = pd.read_csv('beige_book_1996_2025.csv')


# In[37]:


bbdf.head()


# In[38]:


negative_scores = []
positive_scores = []
uncertainty_scores = []
for t in bbdf['text']:
    doc_split = t.split(' ')
    score = sentiment_dict.calculate_sentiment_score(doc_split)
    negative_scores.append(score['negative'])
    positive_scores.append(score['positive'])
    uncertainty_scores.append(score['uncertainty'])


# In[39]:


bbdf['negative_score'] = negative_scores
bbdf['positive_score'] = positive_scores
bbdf['uncertainty_score'] = uncertainty_scores


# In[40]:


bbdf.head()


# In[31]:


max(bbdf['negative_score'])


# In[32]:


min(bbdf['negative_score'])


# In[33]:


print(max(bbdf['positive_score']))
print(min(bbdf['positive_score']))


# In[ ]:


print(max(bbdf['uncertainty_score']))
print(min(bbdf['uncertainty_score']))


# In[41]:


bbdf.to_csv("beige_book_sentiment_scores_1996_2025.csv")

