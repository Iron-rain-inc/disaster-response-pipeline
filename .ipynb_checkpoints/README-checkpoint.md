# Disaster Response Pipeline Project

This project is an ETL and ML pipeline for analysis of disaster response messages. 

## Getting Started

To get started training with this project you will need to obtain a disaster messages and a disasters categories csv file in the format:
Messages - id, message, original, genre
Categories - id, categories. Where categories is of the format 'category-0|1;category-0|1'

To get started using the model follow the instructions for running and browse to the directed address. You will be able to categorize your message. 

### Projcet Motivation
This project is designed to assist in the event of an emergency by classifying messages into the appropriate categories. 
From this state an invocation system could be implemented to automatically forward alerts and messages as required. 

### Instructions for running:
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


### Project Notes

#### Algorithms 
The following algorithms are used in this pipeline:
- CountVectorizer
    - The count vectorizer converts collections of text in a token matrix for for modelling. 
    
- TfidfTransformer
    - The tfidf is a token frequency identifier that automatically takes care of common words such as; the, am, and. These words are used very frequently and will skew the model if left in. 
    
- MultiOutputClassifier    
    - OneVsRestClassifier
    - SGDClassifier
        - The purpose of the multi output classifier is to extend the one vs rest calssifier to support multi target classification. 