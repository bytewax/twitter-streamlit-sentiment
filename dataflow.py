import re
from datetime import timedelta
from collections import defaultdict

from bytewax.dataflow import Dataflow
from bytewax.inputs import ManualInputConfig
from bytewax.outputs import StdOutputConfig, ManualOutputConfig
from bytewax.execution import run_main
from bytewax.window import TumblingWindowConfig, SystemClockConfig

import spacy
from transformers import AutoTokenizer, AutoModelForSequenceClassification, AutoConfig
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from scipy.special import softmax
import pandas as pd

from contractions import contractions
from twitter import get_rules, delete_all_rules, get_stream, set_stream_rules

# load spacy stop words
en = spacy.load('en_core_web_sm')
en.Defaults.stop_words |= {"s","t",}
sw_spacy = en.Defaults.stop_words

# load contractions and compile regex
pattern = re.compile(r'\b(?:{0})\b'.format('|'.join(contractions.keys())))

# load sentiment analysis model
MODEL = "model/"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)
config = AutoConfig.from_pretrained(MODEL)

# window size in minutes
WINDOW_SIZE = 1

# set up page details
st.set_page_config(
    page_title="Live Twitter Sentiment Analysis",
    page_icon="🐝",
    layout="wide",
)


def input_builder(worker_index, worker_count, resume_state):
    return get_stream()


def remove_username(tweet):
    """
    Remove all the @usernames in a tweet
    :param tweet:
    :return: tweet without @username
    """
    return re.sub('@[\w]+', '', tweet)


def clean_tweet(tweet):
    """
    Removes spaces and special characters to a tweet
    :param tweet:
    :return: clean tweet
    """
    tweet = tweet.lower()
    tweet = re.sub(pattern, lambda g: contractions[g.group(0)], tweet)
    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)", " ", tweet).split())


def get_tweet_sentiment(tweet):
    """
    Determines the sentiment of a tweet whether positive, negative or neutral
    :param tweet:
    :return: sentiment and the tweet
    """
    encoded_input = tokenizer(tweet, return_tensors='pt')
    output = model(**encoded_input)
    scores = output[0][0].detach().numpy()
    scores = softmax(scores)
    ranked = np.argsort(scores)
    ranked = ranked[::-1]
    sentiment_class = config.id2label[ranked[0]]
    sentiment_score = scores[ranked[0]]

    return sentiment_class, tweet


def output_builder1(worker_index, worker_count):
    con = st.empty()
    def write_tweets(sentiment__tweet):
        sentiment, tweet = sentiment__tweet
        con.write(f'sentiment:{sentiment}, tweet:{tweet}')

    return write_tweets


def split_text(sentiment__text):
    sentiment, text = sentiment__text
    tokens = re.findall(r'[^\s!,.?":;0-9]+', text)
    data = [(sentiment, word) for word in tokens if word not in sw_spacy]
    return data

# Add a fold window to capture the count of words
# grouped by positive, negative and neutral sentiment
cc = SystemClockConfig()
wc = TumblingWindowConfig(length=timedelta(minutes=WINDOW_SIZE))

def count_words():
    return defaultdict(lambda:0)


def count(results, word):
    results[word] += 1
    return results


def sort_dict(key__data):
    key, data = key__data
    return ("all", {key: sorted(data.items(), key=lambda k_v: k_v[1], reverse=True)})


def join(all_words, words):
    all_words = dict(all_words, **words)
    return all_words


def join_complete(all_words):
    return len(all_words) == 3


def output_builder2(worker_index, worker_count):
    placeholder = st.empty()
    def write_to_dashboard(key__data):
        key, data = key__data
        with placeholder.container():
            fig, axes = plt.subplots(1, 3)
            i = 0
            for sentiment, words in data.items():
                # Create and generate a word cloud image:
                wc = WordCloud().generate(" ".join([" ".join([x[0],]*x[1]) for x in words]))

                # Display the generated image:
                axes[i].imshow(wc)
                axes[i].set_title(sentiment)
                axes[i].axis("off")
                axes[i].set_facecolor('none')
                i += 1
            st.pyplot(fig)

    return write_to_dashboard

if __name__ == "__main__":

    st.title("Twitter Analysis")

    flow = Dataflow()
    flow.input("input", ManualInputConfig(input_builder))
    flow.map(remove_username)
    flow.map(clean_tweet)
    flow.inspect(print)
    flow.map(get_tweet_sentiment)
    flow.inspect(print)
    flow.capture(ManualOutputConfig(output_builder1))
    flow.flat_map(split_text)
    flow.fold_window(
        "count_words", 
        cc, 
        wc, 
        builder = count_words, 
        folder = count)
    flow.map(sort_dict)
    flow.reduce("join", join, join_complete)
    flow.inspect(print)
    flow.capture(ManualOutputConfig(output_builder2))

    search_terms = [st.text_input('Enter your search terms')]
    print(search_terms)

    if st.button("Click to Start Analyzing Tweets"):
        rules = get_rules()
        delete = delete_all_rules(rules)
        set_stream_rules(search_terms)
        run_main(flow)
