#!/usr/bin/env python
# coding: utf-8

# # First Model

# What we've figured out:
# - SMOTE: stick to 'minority' rather than 'not majority' setting, keep k value the same
# - We've fixed hyperparameters (batch size, epochs, LR), model architecture (num layers, dims, dropout), features (rolling window size), train-val-test split sizes

# In[7]:


import pandas as pd
import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
from scipy.sparse import hstack
from torch import nn
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
from imblearn.over_sampling import SMOTE


# In[8]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)


# ## Collect the data

# In[9]:


# read the csvs
beige_df = pd.read_csv("data/beige_book_1996_2025.csv")
labels_df = pd.read_csv("data/filtered_labels.csv")
lm_sentiment_df = pd.read_csv('data/beige_book_sentiment_scores_1996_2025.csv')
market_futures_df = pd.read_csv('data/futures_with_diff.csv')

# column to merge on is timestamp
labels_df['timestamp'] = labels_df['observation_date']
market_futures_df['timestamp'] = market_futures_df['exp_date']
beige_df = pd.merge(beige_df, labels_df, on='timestamp', how='inner')
beige_df = pd.merge(beige_df, lm_sentiment_df, on='timestamp', how='inner')
beige_df = pd.merge(beige_df, market_futures_df, on='timestamp', how='inner')

# process the columns
beige_df['text'] = beige_df['text_x']
beige_df['url'] = beige_df['url_x']
beige_df['month'] = beige_df['month_x']
beige_df['year'] = beige_df['year_x']
beige_df = beige_df.drop(['year_x', 'month_x', 'url_x', 'text_x'], axis=1)
beige_df = beige_df.drop(['Unnamed: 0', 'Unnamed: 0_x', 'Unnamed: 0_y', 'observation_date', 'exp_date'], axis=1)
beige_df = beige_df.drop(['year_y', 'month_y', 'url_y', 'text_y'], axis=1)

# add average for new columns
beige_df['negative_score_10_mean'] = beige_df['negative_score'].rolling(window=24, min_periods=1).mean().shift(1)
beige_df['positive_score_10_mean'] = beige_df['positive_score'].rolling(window=24, min_periods=1).mean().shift(1)
beige_df['uncertainty_score_10_mean'] = beige_df['uncertainty_score'].rolling(window=24, min_periods=1).mean().shift(1)
beige_df['futures_price_10_mean'] = beige_df['futures_price'].rolling(window=24, min_periods=1).mean().shift(1)
beige_df['implied_rate_10_mean'] = beige_df['implied_rate'].rolling(window=24, min_periods=1).mean().shift(1)
beige_df['implied_rate_diff_10_mean'] = beige_df['implied_rate_diff'].rolling(window=24, min_periods=1).mean().shift(1)

beige_df = beige_df.drop(index=0) # first row is na

# sort by timestamp
final_df = beige_df.sort_values(by='timestamp')

print(len(final_df))
print(final_df.columns)
final_df.head()


# # Feature Engineering

# In[10]:


# standardize the labels (so numeric labels of 0, 1, 2) - raise: 2, hold: 0, lower: 1
label_encoder = LabelEncoder()
final_df['labels'] = label_encoder.fit_transform(final_df['decision'])


# In[11]:


# train-test split (70-15-15 split)
X = final_df[['timestamp', 'rate', 'negative_score', 'positive_score',
        'uncertainty_score', '52W_high', '52W_low', '52W_pct_chg',
        'futures_price', 'implied_rate', 'implied_rate_diff', 'text', 'url',
        'month', 'year', 'negative_score_10_mean', 'positive_score_10_mean',
        'uncertainty_score_10_mean', 'futures_price_10_mean',
        'implied_rate_10_mean', 'implied_rate_diff_10_mean']]
y = final_df['labels']

X_train, X_temp, y_train, y_temp = train_test_split(X, y, random_state=seed, test_size=0.3, shuffle=True)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, random_state=seed, test_size=0.5, shuffle=True)

print(len(X_train), len(X_val), len(X_test))


# In[12]:


# generate TF-IDF encodings for the text
tf_idf_vectorizer = TfidfVectorizer()
X_text_train = tf_idf_vectorizer.fit_transform(X_train['text'])
X_text_val = tf_idf_vectorizer.transform(X_val['text'])
X_text_test = tf_idf_vectorizer.transform(X_test['text'])

print(X_text_train.shape, X_text_val.shape, X_text_test.shape)


# In[13]:


scaler = StandardScaler()
columns_to_scale = ['rate', 'negative_score', 'positive_score', 'uncertainty_score', '52W_high', '52W_low', '52W_pct_chg', 
        'futures_price', 'implied_rate', 'implied_rate_diff', 'negative_score_10_mean', 'positive_score_10_mean',
        'uncertainty_score_10_mean', 'futures_price_10_mean', 'implied_rate_10_mean', 'implied_rate_diff_10_mean']

X_nums_train = scaler.fit_transform(X_train[columns_to_scale])
X_nums_val   = scaler.transform(X_val[columns_to_scale])
X_nums_test  = scaler.transform(X_test[columns_to_scale])

print(X_nums_train.shape, X_nums_val.shape, X_nums_test.shape)


# In[14]:


X_train = hstack([X_nums_train, X_text_train])
X_val = hstack([X_nums_val, X_text_val])
X_test = hstack([X_nums_test, X_text_test])

print(X_train.shape, X_val.shape, X_test.shape)


# In[15]:


# SMOTE stuff
smote = SMOTE(sampling_strategy='minority', random_state=seed)
X_train, y_train = smote.fit_resample(X_train, y_train)


# ## Make the model

# In[16]:


class FirstExperimentModel(nn.Module):
    def __init__(self, input_dim):
        super().__init__()

        # input layer, starting with 512 since i've not seen 1024 before
        self.input_layer = nn.Sequential(nn.Linear(input_dim, 256), nn.ReLU(), nn.Dropout(0.3))

        # hidden layers, reducing dims further
        self.hidden_layer = nn.Sequential(nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.3))
        self.hidden_layer_2 = nn.Sequential(nn.Linear(128, 32), nn.ReLU(), nn.Dropout(0.3))

        # output layer without softmax
        self.output_layer = nn.Linear(32, 3)

        # all together now
        self.net = nn.Sequential(self.input_layer, self.hidden_layer, self.hidden_layer_2, self.output_layer)
    
    def forward(self, x):
        return self.net(x)
    
model = FirstExperimentModel(X_train.shape[1])
model.to(device)
print(model.net)


# In[17]:


# define cost-sensitive loss function
# inspiration: https://medium.com/nerd-for-tech/review-cb-loss-class-balanced-loss-based-on-effective-number-of-samples-image-classification-3056a1a1a001
weights = 1.0 / np.bincount(y_train.values)
weights = weights / weights.sum()
weights = torch.tensor(weights, dtype=torch.float32).to(device)
criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, device=device))

lr = 0.0001 # making this v small since we have less data
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


# ## Training Loop

# TODO: figure out how to save the best model by tracking the best validation error seen per epoch

# In[18]:


num_epochs = 50 # idk apparently for tf-idf you need large epochs
batch_size = 32 # this is sort of random (smaller might work better since we have less data to begin with)


# In[19]:


# heavily copied from HW4

def train(data_loader, model, criterion, optimizer, device):
    train_avg_loss = 0
    num_correct = 0
    all_preds = []
    all_labels = []

    model.train()
    for feats, labels in data_loader:
        optimizer.zero_grad()
        feats = feats.to(device)
        labels = labels.long().to(device)

        # Perform forward pass and calculate avg loss over all time steps
        logits = model(feats)
        loss = criterion(logits, labels)
        preds = torch.argmax(logits, dim=1)
        num_correct += torch.sum(preds == labels)
        all_preds.append(preds.cpu())
        all_labels.append(labels.cpu())

        # Backward pass, update weights
        loss.backward()
        optimizer.step()

        train_avg_loss += loss.item()*feats.shape[0]

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    accuracy = num_correct.item() / len(data_loader.dataset)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"Train loss: {train_avg_loss/len(data_loader.dataset)} | Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1: {f1}")
    return train_avg_loss/len(data_loader.dataset)

def val(data_loader, model, criterion, device):
    val_avg_loss = 0
    num_correct = 0
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for feats, labels in data_loader:
            feats = feats.to(device)
            labels = labels.long().to(device)

            # Perform forward pass and calculate avg loss over all time steps
            logits = model(feats)
            loss = criterion(logits, labels)
            preds = torch.argmax(logits, dim=1)
            num_correct += torch.sum(preds == labels)
            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            val_avg_loss += loss.item()*feats.shape[0]

    y_true = torch.cat(all_labels).numpy()
    y_pred = torch.cat(all_preds).numpy()
    accuracy = num_correct.item() / len(data_loader.dataset)
    precision = precision_score(y_true, y_pred, average="macro", zero_division=0)
    recall = recall_score(y_true, y_pred, average="macro", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    print(f"Val loss: {val_avg_loss/len(data_loader.dataset)} | Accuracy: {accuracy} | Precision: {precision} | Recall: {recall} | F1: {f1}\n")
    return val_avg_loss/len(data_loader.dataset)


# In[20]:


X_train_dense = torch.tensor(X_train.toarray(), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.long)
train_dataset = TensorDataset(X_train_dense, y_train_tensor)

X_val_dense = torch.tensor(X_val.toarray(), dtype=torch.float32)
y_val_tensor = torch.tensor(y_val.values, dtype=torch.long)
val_dataset = TensorDataset(X_val_dense, y_val_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

train_loss_list, val_loss_list = [], []
for epoch in range(num_epochs):
    print(f"Epoch {epoch}")
    train_loss_list.append(train(train_dataloader, model, criterion, optimizer, device))
    val_loss_list.append(val(val_dataloader, model, criterion, device))

#with open('model_data/scores.txt', 'a') as f:
#    f.write(f"{batch_size}-{num_epochs}-{lr} | {train_loss_list} | {val_loss_list}\n")


# In[21]:


plt.figure(figsize=(8,5))
plt.plot(train_loss_list, label="Train Loss")
plt.plot(val_loss_list, label="Validation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.show()


# In[22]:


def actual_vs_pred(data_loader, model, device):
    all_preds = []
    all_labels = []

    model.eval()
    with torch.no_grad():
        for feats, labels in tqdm(data_loader):
            feats = feats.to(device)
            labels = labels.long().to(device)

            # Perform forward pass
            logits = model(feats)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return pd.DataFrame({"actual": all_labels, "predicted": all_preds})

actual_vs_pred(val_dataloader, model, device)


# ## Test dataset

# In[23]:


X_test_dense = torch.tensor(X_test.toarray(), dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.long)
test_dataset = TensorDataset(X_test_dense, y_test_tensor)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
results = actual_vs_pred(test_dataloader, model, device)


# In[24]:


print(f"Test Accuracy: {accuracy_score(results["actual"], results["predicted"])}")
print(f"Test Precision: {precision_score(results["actual"], results["predicted"], average="macro")}")
print(f"Test Recall: {recall_score(results["actual"], results["predicted"], average="macro")}")
print(f"Test f1: {f1_score(results["actual"], results["predicted"], average="macro")}")
print(f"Test conf_matrix:\n{confusion_matrix(results["actual"], results["predicted"])}")


# In[35]:


# step 7: evaluation
precision = precision_score(results["actual"], results["predicted"], average="macro")
recall = recall_score(results["actual"], results["predicted"], average="macro")
accuracy = accuracy_score(results["actual"], results["predicted"])
f1 = f1_score(results["actual"], results["predicted"], average="macro")
print("precision", precision)
print("recall", recall)
print("accuracy", accuracy)
print(precision, recall, accuracy)
print(confusion_matrix(results["actual"], results["predicted"]))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
class_names = ["hold", "lower", "raise"]

# Compute confusion matrix
cm = confusion_matrix(results["actual"], results["predicted"])

# Create figure
plt.figure(figsize=(6, 5))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=class_names,
    yticklabels=class_names,
    annot_kws={"size": 20}
)


plt.xticks(fontsize=16)
plt.yticks(fontsize=16, rotation=0)

plt.xlabel("Predicted Label", fontsize=16)
plt.ylabel("True Label", fontsize=16)
plt.title("Best Model Confusion Matrix", fontsize=18)

# Save the figure
plt.tight_layout()
plt.savefig("best_matrix.png", dpi=300)

plt.show()


metrics = ["Accuracy", "Precision", "Recall", "F1"]
metric_values = []

acc = accuracy
prec = precision
rec = recall

f1 = f1
metric_values.append([acc, prec, rec, f1])

metric_values = np.array(metric_values)  # shape: (num_models, 4)

x = np.arange(len(metrics))
width = 0.2  # thicker bars

plt.figure(figsize=(7, 6))
plt.bar(x, metric_values[0], width, label="Best Model", color='tab:blue')

plt.xticks(x, metrics, fontsize=18)  # bigger x-axis labels
plt.yticks(fontsize=14)              # bigger y-axis labels
plt.ylabel("Score", fontsize=18)
plt.title("Performance Metrics For Best Model", fontsize=20)
plt.legend(fontsize=14)
plt.ylim(0, 1)
plt.grid(axis='y')

# optionally, add value labels on top of bars
for i, v in enumerate(metric_values[0]):
    plt.text(x[i], v + 0.02, f"{v:.2f}", ha='center', fontsize=14)

plt.savefig('Best Model Performance Metrics.pdf')
plt.show()


# ## SHAP

# In[19]:


background_inputs = []
for feats, _ in train_dataloader:
    background_inputs.append(feats)
background_data = torch.cat(background_inputs, dim=0).to(device)

test_inputs = []
for feats, _ in test_dataloader:
    test_inputs.append(feats)
test_data = torch.cat(test_inputs, dim=0).to(device)

explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(test_data)

numeric_feature_names = columns_to_scale
tfidf_feature_names = tf_idf_vectorizer.get_feature_names_out().tolist()
feature_names = numeric_feature_names + tfidf_feature_names
print(feature_names)


# In[20]:


shap_values.shape # (num test samples x num features x num classes)


# In[21]:


# axes swapping: https://medium.com/@muneebhashmi10/explainable-convolutional-neural-networks-with-pytorch-shap-62ffb229a918
def top_features_per_class(shap_values, feature_names, top_k):
    N, F, C = shap_values.shape
    
    results = {}
    for cls in range(C):
        vals = shap_values[:, :, cls]
        mean_abs = np.mean(np.abs(vals), axis=0)  # shape (F,)
        idx = np.argsort(-mean_abs)[:top_k]
        results[cls] = list(zip(np.array(feature_names)[idx], mean_abs[idx]))
    
    return results

top_feats = top_features_per_class(shap_values, feature_names, 20)

for cls, feats in top_feats.items():
    print(f"\nTop features for class {cls}:")
    for f, score in feats:
        print(f"{f:25s} {score:.6f}")


# In[32]:


# raise: 2, hold: 0, lower: 1
shap.summary_plot(shap_values[:, :, 0], test_data.cpu().numpy(), feature_names=feature_names, max_display=5)


# In[33]:


# raise: 2, hold: 0, lower: 1
shap.summary_plot(shap_values[:, :, 1], test_data.cpu().numpy(), feature_names=feature_names, max_display=5)


# In[34]:


shap.summary_plot(shap_values[:, :, 2], test_data.cpu().numpy(), feature_names=feature_names, max_display=5)


# In[30]:


shap.plots.bar(shap.Explanation(values=shap_values[:, :, 0], data=test_data.cpu().numpy(), feature_names=feature_names), max_display=10)


# In[31]:


shap.plots.bar(shap.Explanation(values=shap_values[:, :, 1], data=test_data.cpu().numpy(), feature_names=feature_names), max_display=10)


# In[29]:


shap.plots.bar(shap.Explanation(values=shap_values[:, :, 2], data=test_data.cpu().numpy(), feature_names=feature_names), max_display=10)

