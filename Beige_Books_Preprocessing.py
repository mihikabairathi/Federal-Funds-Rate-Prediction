#!/usr/bin/env python
# coding: utf-8

# # Beige Books Data

# This notebook does a few things:
# - Downloads all the Beige Books summaries
# - Validates the data isn't sus
# - Cleans up the text

# ## Imports

# In[83]:


from bs4 import BeautifulSoup
from collections import Counter
import nltk
import pandas as pd
import re
import requests
import string
import time


# In[22]:


nltk.download("stopwords")
nltk.download('punkt_tab')


# ## Download the data

# In[2]:


def fetch_beigebook(url, year, month):
    data = []
    
    try:
        r = requests.get(url, timeout=10)
        if r.status_code != 200:
            return []
        
        soup = BeautifulSoup(r.text, "html.parser")
        paragraphs = []
        
        for p in soup.find_all("p"):
            parents = [parent.name for parent in p.parents]
            if not any(tag in parents for tag in ["header", "footer", "nav"]):
                text = p.get_text(strip=True)
                if text.lower().startswith("full report"):
                    break
                if text:
                    paragraphs.append(text)
        
        if paragraphs:
            combined = " ".join(paragraphs)
            data.append({
                "year": year,
                "month": month,
                "url": url,
                "text": combined
            })
            print(f"Fetched {year}-{month}, {len(paragraphs)} paragraphs")
        
        time.sleep(0.5)
        return data
    
    except Exception as e:
        print(f"Failed to fetch {url}: {e}")
        return []


# In[ ]:


beige_books_data = []

# 1996 - 2010
for year in range(1996, 2011):
    for month in range(1, 13):
        for day in range(1, 32):
            url = f"https://www.federalreserve.gov/fomc/beigebook/{year}/{year}{month:02d}{day:02d}/default.htm"
            beige_books_data.extend(fetch_beigebook(url, year, month))

# 2011 - 2016
for year in range(2011, 2017):
    for month in range(1, 13):
        url = f"https://www.federalreserve.gov/monetarypolicy/beigebook/beigebook{year}{month:02d}.htm?summary"
        beige_books_data.extend(fetch_beigebook(url, year, month))

# 2017 - 2025
for year in range(2017, 2026):
    for month in range(1, 13):
        url = f"https://www.federalreserve.gov/monetarypolicy/beigebook{year}{month:02d}-summary.htm"
        beige_books_data.extend(fetch_beigebook(url, year, month))

# Save to CSV
df = pd.DataFrame(beige_books_data)
df.to_csv("data/beige_book_1996_2025.csv", index=False)
print("Saved beige_book_1996_2025.csv")


# ## Data Validation

# In[137]:


beige_df = pd.read_csv("data/beige_book_1996_2025.csv")


# In[138]:


# assume the first day of the month for timestamp
beige_df["timestamp"] = pd.to_datetime(beige_df["year"].astype(str) + "-" + beige_df["month"].astype(str) + "-01")


# In[139]:


# check for nulls and data types
print(f"Number of rows: {len(beige_df)}")
beige_df.info()


# In[140]:


# see samples of data
print(beige_df.loc[0, "url"])
print(beige_df.loc[230, "text"])
beige_df.head()


# In[141]:


# make sure all the data has been collected
print(f"earliest = {beige_df["timestamp"].min()}")
print(f"latest = {beige_df["timestamp"].max()}")
beige_df.groupby("year")["timestamp"].agg(["count", "min", "max"])


# ## Text Cleaning

# We perform basic case normalization, whitespace cleaning, boilerplate removal. Notably, we don't perform the following:
# - stopword removal (needed for sentiment analysis down the road)
# - punctuation cleaning (gets rid of financial symbols such as currency, percentages, ratios)
# - lemmatization (removes specificity of financial terms)

# In[142]:


# case normalization
beige_df["text"] = beige_df["text"].str.lower()


# In[143]:


# clean whitespace
beige_df["text"] = beige_df["text"].str.strip()
beige_df["text"] = beige_df["text"].str.replace(r"\s+", " ", regex=True)


# In[144]:


# boilerplate removal
def remove_prefixes(text):
    prefixes = [
        "and is not a commentary on the views of federal reserve officials.", 
        "and is not a comment on the views of federal reserve officials", 
        "and is not a representation of the views of federal reserve officials.",
        "and is not a commentary on the views of the federal reserve officials.",
        "and is not a commentary of the views of federal reserve officials.",
        "share sensitive information only on official, secure websites."
    ]
    prefix_regex = re.compile("|".join(prefixes))
    
    match = prefix_regex.search(text)
    new_text = text[match.end():].strip() if match else text

    if len(text) - len(new_text) > 600:
        print(f"Warning: removed prefix length {len(text) - len(new_text)} exceeds threshold")
    return new_text

def remove_suffixes(text):
    suffixes = [
        "return to topbostonhome", 
        "summary districtsbostonnew", 
        "return to top this page uses javascript.",
        "note: this report was prepared at the federal reserve bank of"
    ]
    suffix_regex = re.compile("|".join(suffixes))
    
    match = suffix_regex.search(text)
    new_text = text[:match.start()].strip() if match else text

    if len(text) - len(new_text) > 400:
        print(f"Warning: removed prefix length {len(text) - len(new_text)} exceeds threshold")
    return new_text

beige_df["text"] = beige_df["text"].apply(remove_prefixes)
beige_df["text"] = beige_df["text"].apply(remove_suffixes)


# In[145]:


# confirm that we don't need to do chunking since the summary lengths are relatively small
print("Number of words:")
beige_df["text"].str.split().apply(len).agg(["min", "max", "mean", "median"])


# In[146]:


# unigrams and bigrams
all_tokens = []
stopwords = set(nltk.corpus.stopwords.words("english"))

for text in beige_df["text"]:
    all_tokens.extend([word for word in nltk.word_tokenize(text) if word not in stopwords and word not in string.punctuation])

print(f"Top unigrams: {Counter(all_tokens).most_common(10)}")
print(f"Top bigrams: {Counter(nltk.ngrams(all_tokens, 2)).most_common(10)}")


# # Write Final Data

# In[147]:


beige_df.to_csv("data/beige_book_1996_2025.csv", index=False)

