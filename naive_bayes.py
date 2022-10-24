from distutils.log import error
import nltk
import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
import pandas as pd
from nltk.corpus import twitter_samples 

nltk.download('twitter_samples')
nltk.download('stopwords')


all_positive_tweets = twitter_samples.strings('positive_tweets.json')
all_negative_tweets = twitter_samples.strings('negative_tweets.json')

test_pos = all_positive_tweets[4000:]
train_pos = all_positive_tweets[:4000]
test_neg = all_negative_tweets[4000:]
train_neg = all_negative_tweets[:4000]

train_x = train_pos+train_neg
test_x = test_pos+test_neg

train_y = np.append(np.ones(len(train_pos)),np.zeros(len(train_neg)))
test_y = np.append(np.ones(len(test_pos)),np.zeros(len(test_neg)))

print("train_y.shape = " + str(train_y.shape))
print("test_y.shape = " + str(test_y.shape))


def process_tweet(tweet):

    stemmer = PorterStemmer()
    stopwords_en = stopwords.words('english')

    tweet = re.sub(r'\$\w*','',tweet)
    tweet = re.sub(r'^RT[/s]+','',tweet)
    tweet = re.sub(r'https?:\/\/.*[\r\n]*','',tweet)
    tweet = re.sub(r'#','',tweet)

    tokenizer = TweetTokenizer(preserve_case=False,strip_handles=True,reduce_len=True)

    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []

    for word in tweet_tokens:
        if (word not in stopwords_en and word not in string.punctuation):
            stem_word = stemmer.stem(word)
            tweets_clean.append(stem_word)
    
    return tweets_clean


def count_tweets(result,tweets,ys):

    yslist = np.squeeze(ys).tolist()
    for y,tweet in zip(yslist,tweets):
        for word in process_tweet(tweet):
            pair = (word,y)

            if pair in result:
                result[pair]+=1
            else:
                result[pair]=1
    
    return result


print('Tweet: \n', train_x[0])
print('\nProcessed tweet: \n', process_tweet(train_x[0]))

freqs = count_tweets({}, train_x, train_y)


def lookup(freqs,word,label):
    n=0
    pair=(word,label)
    if (pair in freqs):
        n=freqs[pair]
    
    return n

def train_naive_bayes(freqs,train_x,train_y):
    loglikelihood={}
    logprior=0

    vocab = set([pair[0] for pair in freqs.keys()])
    V=len(vocab)

    N_pos = N_neg = V_pos = V_neg = 0

    for pair in freqs.keys():
        if pair[1]>0:
            V_pos+=1
            N_pos+=freqs[pair]
        else:
            V_neg+=1
            N_neg+=freqs[pair]
    
    D=len(train_y)
    D_pos = (len(list(filter(lambda x:x>0,train_y))))
    D_neg = (len(list(filter(lambda x:x<=0,train_y))))

    logprior = np.log(D_pos)-np.log(D_neg)


    for word in vocab:
        freq_pos = lookup(freqs,word,1)
        freq_neg = lookup(freqs,word,0)

        p_w_pos = (freq_pos + 1) / (N_pos + V)
        p_w_neg = (freq_neg + 1) / (N_neg + V)
        loglikelihood[word] = np.log(p_w_pos/p_w_neg)
    
    return logprior,loglikelihood


def predict(tweet,logprior,loglikelihood):
    word_l = process_tweet(tweet)

    p=0
    p+=logprior

    for word in word_l:
        if word in loglikelihood:
            p+=loglikelihood[word]
    return p

mytweet = 'very bad.'

logprior,loglikelihood=train_naive_bayes(freqs,train_x,train_y)

p = predict(mytweet,logprior,loglikelihood)

print(p)



def test(test_x,test_y,logprior,loglikelihood):
    accuracy=0
    y_pred=[]

    for tweet in test_x:
        if predict(tweet,logprior,loglikelihood)>0:
            y_pred_i=1
        else:
            y_pred_i=0
    
        y_pred.append(y_pred_i)

    error=np.mean(np.absolute(y_pred-test_y))

    accuracy = 1-error

    return accuracy

print("Accuracy=",test(test_x,test_y,logprior,loglikelihood))


