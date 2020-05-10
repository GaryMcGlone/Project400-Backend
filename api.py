## GENERAL
import pandas as pd
import sklearn
import numpy as np
import tweepy
import boto3
from boto3.dynamodb.conditions import Attr

## fLASK
from flask import Flask
from flask import request
from flask import jsonify
from flask_cors import CORS

## NLTK
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

## UTILITY
from joblib import load
import re
import preprocessor as p
import string
import credentials
import json
import collections
import uuid
import datetime
import random

## Dynamno
client = boto3.resource('dynamodb')
table = client.Table('searches')

app = Flask(__name__)
cors = CORS(app)

# Init tweepy
auth = tweepy.OAuthHandler(credentials.CONSUMER_KEY, credentials.CONSUMER_SECRET)
auth.set_access_token(credentials.ACCESS_TOKEN, credentials.ACCESS_TOKEN_SECRET)
api = tweepy.API(auth, wait_on_rate_limit=True)

classifier = load('model_artifacts/model.sav')
tfidf_vectorizer = load('model_artifacts/tfidf_vectorizer.sav')

@app.route('/search', methods=['GET', 'POST'])
def get_tweets():
    search = request.args.get('searchTerm')
    max_tweets = 200
    api = tweepy.API(auth, wait_on_rate_limit=True)
    
    tweets = [status for status in tweepy.Cursor(api.search, q=f'{search} -filter:retweets',lang='en', tweet_mode='extended').items(max_tweets)]
    
    y_predictions = classifier.predict(tfidf_vectorizer.transform(x.full_text for x in tweets))

    data = [{ 'tweet' : tweet.full_text, 'sentiment' : str(sentiment) } for tweet, sentiment in zip(tweets, y_predictions)]

    top_words = word_freq(data)

    table.put_item(Item={ 
        "Positive_Count" : str(sum(y_predictions == 1)),
        "Negative_Count" : str(sum(y_predictions == 0)),
        'uuid': uuid.uuid4().hex,
        "Searched": search,
        "Date" : str(datetime.datetime.now()),
        "Score" : str(round((sum(y_predictions == 1) / len(tweets)) * 100 ,2))
     })       

    # return 4 random classified tweets instead of all 200
    sampling = random.choices(data, k=4)

    return json.dumps({
        "classified_tweets" : sampling,
        "top_words_associated" : top_words
    })


def word_freq(tweets):

    ## CLEAN TWEET
    punct = set(string.punctuation)
    cleaned_tweets = [p.clean(t['tweet']) for t in tweets]

    ## REMOVE PUNCTUATION
    exclude = string.punctuation + "'"
    stripped_clean_tweets = [s.translate(str.maketrans('', '', exclude)) for s in cleaned_tweets]

    ##  REMOVE STOPWORDS
    stripped_clean_tweets_no_sw = remove_stopwords(stripped_clean_tweets)
    string_to_split = ''.join(stripped_clean_tweets_no_sw).lower()

    ## CALCULATE FREQUENCY 
    split_it = string_to_split.split() 
    counter = collections.Counter(split_it) 

    most_occur = counter.most_common(8) 
    return most_occur


def remove_stopwords(data):
    sw=set(stopwords.words('english'))
    output_array=[]
    for sentence in data:
        temp_list=[]
        for word in sentence.split():
            if word.lower() not in sw:
                temp_list.append(word)
        output_array.append(' '.join(temp_list))
    return output_array


@app.route('/search_timeline', methods=['GET', 'POST'])
def search_timeline():
    search = request.args.get('searchTerm')

    response = table.scan(
        FilterExpression=Attr('Searched').eq(search)
    )
        
    items = response['Items']
    
    return json.dumps(items)