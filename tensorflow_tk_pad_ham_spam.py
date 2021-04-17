#%%
#Importing All necessary libraries used in this project

import pandas as pd
import numpy as np
import re
#visualization
import seaborn as sns
import matplotlib.pyplot as plt
#NLP
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import STOPWORDS , WordCloud 
#model
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#Tensorflow
import tensorflow as tf
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import Embedding
from tensorflow.keras.models import Sequential
# %%
df = pd.read_csv("spam.csv")
# %%
df.head(5)
# %%
df=df.drop(["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis=1)

# %%
df.head()
# %%
df=df.rename(columns={"v1":"label" , "v2":"message"})
# %%
df.head()
# %%
df.describe()
# %%
df.groupby('label').describe().T
# %%
# Get all the ham and spam emails
ham_msg = df[df.label =='ham']
spam_msg = df[df.label=='spam']
# %%
ham_msg_text = " ".join(ham_msg.message.to_numpy().tolist())
spam_msg_text = " ".join(spam_msg.message.to_numpy().tolist())
# %%
#WordCloud For the Ham Messages
ham_msg_cloud = WordCloud(width =520, height =260, stopwords=STOPWORDS,max_font_size=50, background_color ="black", colormap='Blues').generate(ham_msg_text)
plt.figure(figsize=(16,10))
plt.imshow(ham_msg_cloud)
plt.show()
# %%
#Worcloud For The Spam Messages
spam_msg_cloud = WordCloud(width =520, height =260, stopwords=STOPWORDS,max_font_size=50, background_color ="black", colormap='Blues').generate(spam_msg_text)
plt.figure(figsize=(16,10))
plt.imshow(spam_msg_cloud)
plt.show()
# %%
plt.figure(figsize=(8,6))
sns.countplot(df.label)

# %%
#percentage of the spam messages
(len(spam_msg)/len(ham_msg))*100
#%%
stopwords = stopwords.words('english')
#%%
df['message']=df['message'].apply(lambda x:re.sub('[.,@#$%*?_]','',x))
#%%
df['message']=df['message'].apply(lambda x:word_tokenize(x))
#%%
df['message']=df['message'].apply(lambda x:[word for word in x if word not in stopwords])
df.head()
# %%
X = df['message']
y = df['label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)
X_train
# %%
X_train = X_train.apply(lambda x :' '.join(x))
# %%
X_train
# %%

t = Tokenizer()
t.fit_on_texts(X_train)
# %%
encoded_train = t.texts_to_sequences(X_train)
encoded_test = t.texts_to_sequences(X_test)
# %%
padded_train = pad_sequences(encoded_train, maxlen=8, padding='post')
padded_test = pad_sequences(encoded_test, maxlen=8, padding='post')
print(padded_test)
#%%
#RandomForest model fitting
model=RandomForestClassifier(n_estimators=100)
model.fit(padded_train,y_train)
#%%
y_pred=model.predict(padded_test)
acc=accuracy_score(y_pred,y_test)
print(acc*100,"%")

# %%
#%%
vocab_size = 50_000
one_hots = [one_hot(word,vocab_size)for word in X_train]
print(one_hots)
# %%
padded = pad_sequences(one_hots, padding ='post' , maxlen=5)
print(padded)
# %%
model = Sequential()
model.add(Embedding(vocab_size,50))
model.compile("adam" , "mse")
# %%
predict = model.predict(padded)
# %%
predict.shape

# %%

# %%
