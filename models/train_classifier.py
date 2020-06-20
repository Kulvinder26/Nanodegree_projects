# import libraries


import sys
sys.path.append("..")


import pandas as pd
from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
import sqlite3
import sys
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier
import pickle

import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from utils.util import tokenize
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_data(database_filepath):
    # load data from database
    engine_str = 'sqlite:///' + database_filepath
    
    engine = create_engine(engine_str)
    df = pd.read_sql_table("messages", engine)
    
    
    X = df['message']
    Y = df.drop(['message','original','genre','id'],axis=1)
    

    cols = Y.columns.tolist()
    
    
    return X,Y,cols



def build_model():
    pipeline = Pipeline([
    
    ('countVec',CountVectorizer(tokenizer=tokenize)),
    ('tfidf', TfidfTransformer()),
    ('clf',MultiOutputClassifier(MultinomialNB(fit_prior =True),n_jobs=-1))
    ])
    
    return pipeline
    


def evaluate_model(model, X_test, Y_test, category_names):
    y_pred = model.predict(X_test)
    
    total_accuracy = (y_pred == Y_test).mean().mean() * 100
    
    
    
    print()
    
    
    
    Y_pred = pd.DataFrame(y_pred, columns=Y_test.columns)
    
    
    for col in Y_test.columns:
        print(classification_report(Y_test[col], Y_pred[col]))
        accuracy = (Y_pred[col].values == Y_test[col].values).mean().mean() * 100
        print('Accuracy:  ',accuracy)
        
    print('Total Accuracy:  ',total_accuracy)

def save_model(model, model_filepath):
    
    with open(model_filepath, 'wb') as f:
        # Pickle the 'model' to disk
        pickle.dump(model, f)
        
    print("Model Saved!!")
    


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()
