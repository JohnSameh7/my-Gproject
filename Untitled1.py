#!/usr/bin/env python
# coding: utf-8

# In[5]:


import os
import re
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from gensim.models import Word2Vec
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

# Specify the paths to the train and test dataset folders
train_folder = "Desktop/7d54rvzxkr-1/all_files_compressed/training"
test_folder = "Desktop/7d54rvzxkr-1/all_files_compressed/testing"

# Function to read data from the XML file
def read_data_from_xml(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        data = file.read()
    return data

# Load the train dataset
train_data = []
train_labels = []
for file_name in os.listdir(train_folder):
    if file_name.endswith(".xml"):
        file_path = os.path.join(train_folder, file_name)
        data = read_data_from_xml(file_path)
        event_match = re.search(r"<event>(.*?)</event>", data)
        sentence_match = re.search(r"<sentence>(.*?)</sentence>", data)
        if event_match and sentence_match:
            event = event_match.group(1)
            sentence = sentence_match.group(1)
            train_data.append(sentence)
            train_labels.append(event)

# Load the test dataset
test_data = []
test_labels = []
for file_name in os.listdir(test_folder):
    if file_name.endswith(".xml"):
        file_path = os.path.join(test_folder, file_name)
        data = read_data_from_xml(file_path)
        event_match = re.search(r"<event>(.*?)</event>", data)
        sentence_match = re.search(r"<sentence>(.*?)</sentence>", data)
        if event_match and sentence_match:
            event = event_match.group(1)
            sentence = sentence_match.group(1)
            test_data.append(sentence)
            test_labels.append(event)


# In[6]:


# Preprocess the text data
def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove non-alphanumeric characters
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return text

X_train = [preprocess_text(text) for text in train_data]
y_train = train_labels
X_test = [preprocess_text(text) for text in test_data]
y_test = test_labels

# Define a custom PyTorch dataset
class EventDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        return text, label


# In[7]:


# Define the deep learning model
class EventModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(EventModel, self).__init__()
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.rnn = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        embedded = self.embedding(x)
        output, _ = self.rnn(embedded)
        output = self.fc(output[:, -1, :])
        return output


# In[8]:


# Set the device for training
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Convert the text data to numerical representation using CountVectorizer
vectorizer = CountVectorizer()
X_train_vectorized = vectorizer.fit_transform(X_train)
X_test_vectorized = vectorizer.transform(X_test)


# In[9]:


# Create DataLoader for training and testing datasets
train_dataset = EventDataset(X_train_vectorized.toarray(), y_train)
test_dataset = EventDataset(X_test_vectorized.toarray(), y_test)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# In[10]:


# Initialize the model and optimizer
model = EventModel(input_size=len(vectorizer.vocabulary_), hidden_size=128, output_size=len(set(y_train)))
model.to(device)
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()


# In[13]:


from torch.utils.data import DataLoader, Dataset
import torch

# Custom dataset class
class MyDataset(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels
        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, index):
        return self.inputs[index], self.labels[index]

# Convert labels to tensors
train_labels_tensor = torch.tensor(train_labels)

# Create the custom dataset
train_dataset = MyDataset(X_train_vectorized, train_labels_tensor)

# Create the dataloader
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Training loop
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    
    for inputs, labels in train_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        train_acc += torch.sum(preds == labels.data)
        
    epoch_loss = train_loss / len(train_dataset)
    epoch_acc = train_acc.double() / len(train_dataset)
    
    print(f"Epoch {epoch+1}/{num_epochs}:")
    print(f"Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}")


# In[ ]:


# Evaluation on the test set
model.eval()
test_loss = 0.0
test_acc = 0.0

all_preds = []
all_labels = []

with torch.no_grad():
    for inputs, labels in test_dataloader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        
        test_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        test_acc += torch.sum(preds == labels.data)
        
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        
test_loss = test_loss / len(test_dataset)
test_acc = test_acc.double() / len(test_dataset)

print(f"Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}")


# In[ ]:


# Classification report
target_names = sorted(set(y_train))
print(classification_report(all_labels, all_preds, target_names=target_names))


# In[ ]:


# Function to preprocess user input
def preprocess_user_input(text):
    text = preprocess_text(text)
    vectorized_text = vectorizer.transform([text]).toarray()
    return torch.from_numpy(vectorized_text).to(device)


# In[ ]:


# Function to extract events from user input using the trained model
def extract_events(text):
    model.eval()
    with torch.no_grad():
        inputs = preprocess_user_input(text)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        return predicted.item()


# In[ ]:


# Example usage
user_input = "This is an example sentence."
predicted_event = extract_events(user_input)
print(f"Predicted Event: {predicted_event}")

