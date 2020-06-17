
# coding: utf-8

# In[15]:


import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from os import path
from PIL import Image
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

import matplotlib.pyplot as plt
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize


# In[2]:


get_ipython().magic('matplotlib inline')


# In[3]:


def load_data(database_filepath):
    # load data from database
    engine_str = 'sqlite:///' + database_filepath
    
    print(engine_str)
    engine = create_engine(engine_str)
    df = pd.read_sql_table("messages", engine)
    
    return df 
    
#    print(df.head(5))
    
#    
    
    
 #   print(Y.head(),Y.columns)
 #   cols = Y.columns.tolist()
    #X,Y,cols
    
   


# In[4]:


df = load_data('data/DisasterResponse.db')


# In[5]:


display(df.head())


# In[6]:


X = df['message']
Y = df.drop(['message','original','genre','id'],axis=1)


# In[36]:


Y.columns.tolist()


# In[46]:


for col in Y.columns.tolist():
    print(col.capitalize())
    text = ','.join(X[Y[col]==1].tolist())

    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    tokens = word_tokenize(text)
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]
    tokens = list(set(tokens))
    clean_text = ','.join(tokens).replace(',',' ')
    if len(clean_text) >10:
        wordcloud = WordCloud(max_font_size=50, max_words=100, background_color="white").generate(clean_text)

        plt.figure(figsize=(8,10))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.savefig(col.capitalize() +'.jpg', bbox_inches='tight')
        plt.show()

