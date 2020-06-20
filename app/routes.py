import json
import plotly
import pandas as pd

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
import joblib
from sqlalchemy import create_engine


from app import app
import os

import sys
sys.path.append("..")

from utils.util import get_tsne_Data

basedir = os.path.abspath(os.path.dirname(__file__))



def tokenize(text):
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens

# load data
basedir = os.path.dirname(basedir)
path = os.path.join(basedir, './data/DisasterResponse.db')

engine = create_engine('sqlite:///' +  path )
df = pd.read_sql_table('messages', engine)

# load model
model = joblib.load(basedir + "/" +  "./models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/index')
def api_running():
    
    return json.dumps({'success': 'success'})



@app.route('/bar')
def bar_graph():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    #print(ids)


    #print()
    #print(graphJSON)
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)



@app.route('/')
def plotting_tsne():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    graphs = get_tsne_Data()

    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    #print(graphJSON)
    # render web page with plotly graphs
    return render_template('master.html', ids = ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


@app.route('/wordclouds')
def wordcloud():

    return render_template('word_clouds.html')

    
#def main():
#    #app.run(host='0.0.0.0', port=3001, debug=True)


#if __name__ == '__main__':
#    main()
