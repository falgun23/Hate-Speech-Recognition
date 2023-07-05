import sklearn
import numpy as np
import pandas as pd
import pickle
import re
import preprocessor as p #pip3 install tweet-preprocessor not pip3 install preprocessor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import svm
from sklearn.metrics import accuracy_score


# training data
train = pd.read_csv("dataset/1.csv", low_memory=False)

# test data
test = pd.read_csv("dataset/test.csv")

# non-racist/sexist related tweets
sum(train["Label"] == 0)

# racist/sexist related tweets
sum(train["Label"] == 1)


# Data cleaning

# remove special characters using the regular expression library
#set up punctuations we want to be replaced
REPLACE_NO_SPACE = re.compile("(\.)|(\;)|(\:)|(\!)|(\')|(\?)|(\,)|(\")|(\|)|(\()|(\))|(\[)|(\])|(\%)|(\$)|(\>)|(\<)|(\{)|(\})")
REPLACE_WITH_SPACE = re.compile("(<br\s/><br\s/?)|(-)|(/)|(:).")

# custum function to clean the dataset (combining tweet_preprocessor and reguar expression)
def clean_tweets(df):
  tempArr = []
  for line in df:
    # send to tweet_processor
    tmpL = p.clean(line)
    # remove puctuation
    tmpL = REPLACE_NO_SPACE.sub("", tmpL.lower()) # convert all tweets to lower cases
    tmpL = REPLACE_WITH_SPACE.sub(" ", tmpL)
    tempArr.append(tmpL)
  return tempArr

# clean training data
train_tweet = clean_tweets(train["tweet"])
train_tweet = pd.DataFrame(train_tweet)

# append cleaned tweets to the training data
train["clean_tweet"] = train_tweet

# clean the test data and append the cleaned tweets to the test data
test_tweet = clean_tweets(test["tweet"])
test_tweet = pd.DataFrame(test_tweet)

# append cleaned tweets to the training data
test["clean_tweet"] = test_tweet


# extract the labels from the train data
y = train.Label.values

# use 70% for the training and 30% for the test
x_train, x_test, y_train, y_test = train_test_split(train.clean_tweet.values, y, 
                                                    stratify=y, 
                                                    random_state=1, 
                                                    test_size=0.3, shuffle=True)


# Vectorize tweets using CountVectorizer

# vectorize tweets for model building
vectorizer = CountVectorizer(binary=True, stop_words='english')

# learn a vocabulary dictionary of all tokens in the raw documents
vectorizer.fit(list(x_train) + list(x_test))

# transform documents to document-term matrix
x_train_vec = vectorizer.transform(x_train)
x_test_vec = vectorizer.transform(x_test)


load_model = pickle.load(open("weights.pkl", 'rb'))

name = input('Enter String: ')
name = [name]

conv = vectorizer.transform(name)

pred = load_model.predict(conv)

if pred == 0:
    print('Non-hate')
elif pred == 1:
    print('Hate')
