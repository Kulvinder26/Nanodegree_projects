
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize
import re	
from numpy import load
import os
import plotly.graph_objs as go
import pandas as pd


def tokenize(text):
    
    stop_words = stopwords.words("english")
    lemmatizer = WordNetLemmatizer()
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    
    # tokenize text
    tokens = word_tokenize(text)
    
    # lemmatize andremove stop words
    tokens = [lemmatizer.lemmatize(word) for word in tokens if word not in stop_words]

    return tokens
    




def get_tsne_Data():

    basedir = os.path.abspath(os.path.dirname(__file__))

    basedir = os.path.dirname(basedir)
    #print(basedir)

    results = load(basedir + '/' +  './data/data.npy')
    classes = pd.read_csv(basedir + '/' + './data/tsne_classes.csv')['Classes']

    unique_classes = classes.unique()


    #print(unique_classes)
    


    graph_one = []

    for i in unique_classes:
        #print(x_val[i],y_val[i])
        #print(classes[i])
        graph_one.append(
              go.Scatter(
              x = results[classes==i,0].tolist(),
              y = results[classes==i,1].tolist(),
              type = 'scatter',
              mode='markers',
              name =  i
            
              )
          )

    layout_one =  {
        "title": "Analyzing Data using a t-SNE Visyualization",
        "xaxis": {
            "title": "x"
        },
        "yaxis": {
            "title": "y"
        },
        "height": 800,
        "showlegend": True
    }

    

    figures = []
    figures.append(dict(data=graph_one, layout=layout_one))


    #print(graph_one)

    return figures
