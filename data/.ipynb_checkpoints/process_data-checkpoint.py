import sys
import pandas as pd
import numpy as np
import re

from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):    
    """
    Takes in the filepath of a message and categories filepath and returns a dataframe.
    messages_filepath - Filepath to messages csv
    categories_filepath - Filepath to categories csv
    """
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    df = pd.concat([messages.set_index('id'), categories.set_index('id')], axis=1, join='inner').reset_index()
    
    categories = df['categories'].str.split(";", expand=True)
    headers = list(categories.iloc[0])
    
    categories.columns = [re.sub('-\d', '', i) for i in headers]
    
    for column in categories:    
        pattern = categories[column].str.extract(r'(\d)', expand=True)
        categories[column] = pattern.astype('int')
    
    df = pd.concat([df, categories], axis=1)
    
    df = df.drop('categories', axis=1)
    
    return df


def clean_data(df):
    """
    Cleans message data, drops duplicates
    df - message dataframe
    """
    df = df.drop_duplicates()
    return df


def save_data(df, database_filename):
    """
    Saves a dataframe to a sql file
    df - dataframe to save
    database_filename - filepath to save the database to
    """
    engine = create_engine('sqlite:///data/DisasterResponse.db')  
    df.to_sql('MessageData', engine, index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()