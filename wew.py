#!/usr/bin/env python
# coding: utf-8

# In[1]:


pip install nltk


# In[1]:


import nltk
nltk.download('reuters')
from nltk.corpus import reuters


# In[2]:


nltk.download('punkt')


# In[3]:


cats = reuters.categories()
print("Reuters has %d categories:\n%s" % (len(cats), cats))


# In[8]:


print(reuters.readme())


# In[4]:


total = len(reuters.paras())
total_multi = 0
for c in cats:
    lc = len(reuters.paras(categories=[c]))
    total_multi += lc
    print("%s ---- %d documents out of %d" % (c, lc, total))
print("Articles belong to %.4f categories on average" % ((total_multi * 1.0) / total))
print("There are %.4f articles per category on average" % ((total * 1.0) / len(cats)))


# In[5]:


from nltk.probability import FreqDist


# In[6]:


fd = FreqDist(reuters.words())


# In[7]:


len(fd)


# In[8]:


import inspect
print(inspect.getargspec(reuters.paras))
print(inspect.getargspec(reuters.fileids))


# In[9]:


reuters.fileids(categories=['yen'])


# In[10]:


reuters.paras(fileids=['test/14913'])


# In[11]:


def isTest(fileid):
    return fileid[:4]=='test'

isTest('test/12345')


# In[12]:


import nltk
from nltk.stem.porter import PorterStemmer

token_dict = {}
stemmer = PorterStemmer()

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed

for file in reuters.fileids():
    if not isTest(file):
        token_dict[file] = stem_tokens(reuters.paras(fileids=[file])[0][0], stemmer)


# In[13]:


stemmer.stem("investigation")


# In[14]:


len(token_dict)


# In[15]:


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score
tfidf = TfidfVectorizer(stop_words='english', input='content')
tfs = tfidf.fit_transform([" ".join(l) for l in token_dict.values()])


# In[16]:


tfs


# In[17]:


for t in tfs[0]:
    print(t)


# In[18]:


tfidf.inverse_transform(tfs[0])


# In[19]:


tfidf


# In[20]:


from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.preprocessing.text import Tokenizer
from keras.utils import np_utils


# In[21]:


max_words = 1000
batch_size = 128
nb_epoch = 1000


# In[22]:


# Load the Reuters dataset
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=max_words, test_split=0.1)


# In[7]:


pip install np


# In[98]:


import numpy as np


# In[99]:


# Convert the data to one-hot encoding
num_classes = np.max(y_train) + 1
y_train = np_utils.to_categorical(y_train, num_classes)
y_test = np_utils.to_categorical(y_test, num_classes)


# In[100]:


# Tokenize the data
tokenizer = Tokenizer(num_words=max_words)
X_train = tokenizer.sequences_to_matrix(X_train, mode='binary')
X_test = tokenizer.sequences_to_matrix(X_test, mode='binary')


# In[107]:


# Build the model
model = Sequential()
model.add(Dense(512, input_shape=(max_words,)))
model.add(Dense(1024, input_shape=(max_words,)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])


# In[112]:


# Train the model
model.fit(X_train, y_train, batch_size=batch_size, epochs=nb_epoch, verbose=1, validation_split=0.1)


# In[114]:


# Evaluate the model on the test data
score = model.evaluate(X_test, y_test, batch_size=batch_size, verbose=1)
print('Test accuracy:', score[1])


# In[25]:


print(len(reuters.fileids(categories=['yen'])))


# In[ ]:




