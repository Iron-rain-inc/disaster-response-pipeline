# Supress scikit warnings
def warn(*args, **kwargs):
        pass
import warnings
warnings.warn = warn

import sys
import nltk
nltk.download(['punkt', 'wordnet', 'averaged_perceptron_tagger'])
import re
import numpy as np
import pandas as pd
import pickle

from sqlalchemy import create_engine
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.multioutput import MultiOutputClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer

# Pactools is used for progress bar during training models 
from pactools import simulate_pac
from pactools.grid_search import ExtractDriver, AddDriverDelay
from pactools.grid_search import DARSklearn, MultipleArray
from pactools.grid_search import GridSearchCVProgressBar



def load_data(database_filepath):
    """
    Load data from a database and output an X,Y dataframe as well as labels
    database_filepath - Filepath to the database to be loaded
    """
    database_filepath = 'sqlite:///' + database_filepath    
    
    engine = create_engine(database_filepath)
    df = pd.read_sql(sql="SELECT * FROM MessageData", con=engine)
    
    X = df['message'].values
    Y = df.drop(['id','message','original','genre'], axis=1)
    category_names = Y.columns.values
    
    return X, Y, category_names


def tokenize(text):
    """
    Text tokenizer to use with vect function    
    """
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    clean_tokens = []
    
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    return clean_tokens


def build_model():    
    """
    Builds a model, runs through paramaters with GridSearch and loads the best set
    """
    pipeline = Pipeline([
        ('features', FeatureUnion([
            ('text_pipeline', Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer())
            ])),
        ])),
         ('clf', MultiOutputClassifier(OneVsRestClassifier(SGDClassifier())))
    ])
    
    parameters = {
        'features__text_pipeline__vect__ngram_range': ((1,1),(1,2),(1,3)),
        'features__text_pipeline__tfidf__use_idf': (True, False),
        'features__text_pipeline__tfidf__smooth_idf': (True, False),
        'features__transformer_weights': (
            {'text_pipeline': 1},
            {'text_pipeline': 0.5},
            {'text_pipeline': 0.2}
        ),
        'clf__estimator__estimator__n_jobs': [50],# 100, 200],
        'clf__estimator__estimator__alpha': [0.0001] #0.001,0.01]    
    }


    cv = GridSearchCVProgressBar(pipeline, param_grid=parameters, n_jobs=-1)

    return cv



def evaluate_model(model, X_test, Y_test, category_names):
    """
    Evaluates the model, determines model accuracy per category
    model - trained model 
    X_test - testing data subset
    Y_test - testing data subset
    category_names - names to be used for labels
    """
    y_pred = model.predict(X_test)
    
    labels = category_names
    
    accuracy = (y_pred == Y_test).mean()
    
    print("Labels:", labels)
    
    print("Accuracy:", accuracy)    


def save_model(model, model_filepath):
    """
    Saves the model to pickle file for use at a later date
    """
    pickle.dump(model, open(model_filepath, 'wb'))


def main():    
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)        
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        
        fs = 200.  # Hz
        high_fq = 50.0  # Hz
        low_fq = 5.0  # Hz
        low_fq_width = 1.0  # Hz

        n_epochs = 3
        n_points = 10000
        noise_level = 0.4

        low_sig = np.array([
            simulate_pac(n_points=n_points, fs=fs, high_fq=high_fq, low_fq=low_fq,
                         low_fq_width=low_fq_width, noise_level=noise_level,
                         random_state=i) for i in range(n_epochs)
        ])
        
        X = MultipleArray(low_sig, None)
        
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