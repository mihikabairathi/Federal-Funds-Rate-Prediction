# %% [markdown]
# # Baseline Model
# 

# %% [markdown]
# ## Imports and Constants

# %%
seed = 42
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

# %%
# step 1: Read the csv
beige_df = pd.read_csv("beige_book_1996_2025.csv")
labels_df = pd.read_csv("filtered_labels.csv")
labels_df['timestamp'] = labels_df['observation_date']

beige_df = pd.merge(beige_df, labels_df, on='timestamp', how='inner')
beige_df = beige_df.drop('Unnamed: 0', axis=1)
beige_df = beige_df.drop('observation_date', axis=1)
beige_df = beige_df.drop(index=0) # first row is na

print(len(beige_df))
beige_df.head()

# %%
# step 2: One-hot encode the labels (so numeric labels of 0, 1, 2)
label_encoder = LabelEncoder()
beige_df['labels'] = label_encoder.fit_transform(beige_df['decision'])

# %%
# step 3: Generate TF-IDF encodings for the text
tf_idf_vectorizer = TfidfVectorizer()
X = tf_idf_vectorizer.fit_transform(beige_df['text'])
y = beige_df['labels']
print(X.shape, len(y))

# %%
# step 4: Train-test split (80-20 split)
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=seed, test_size=0.2)

# %%
# step 5: Run logistic regression on train set
model = LogisticRegression(random_state=seed)
model.fit(X_train, y_train)

# %%
# step 6: Run the model on the test set for prediction
X_pred = model.predict(X_train)
y_pred = model.predict(X_test)

# %%
# step 7: evaluation
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)
print(precision, recall, accuracy)
print(confusion_matrix(y_test, y_pred))

# %%
beige_df['decision'].value_counts()

# %%
y_test.value_counts()

# %%
y_train.value_counts()


