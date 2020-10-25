# import pandas as pd
# import numpy as np
#
# df = pd.read_csv("train.csv")
# df['comment_text'] = df['comment_text'].str.lower()
# pass


import sys, os, re, csv, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, Activation
from keras.layers import Bidirectional, GlobalMaxPool1D
from keras.models import Model
from keras import initializers, regularizers, constraints, optimizers, layers


def load_csv():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test


def main():
    max_length = 120
    oov_tok = "<OOV>"
    max_features = 20000
    embedding_dim = 16

    train, test = load_csv()

    list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    y = train[list_classes].values
    list_sentences_train = train["comment_text"]
    list_sentences_test = test["comment_text"]

    tokenizer = Tokenizer(num_words=max_features, oov_token=oov_tok)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

    list_tokenized_train = pad_sequences(list_tokenized_train, maxlen=max_length, truncating='post')
    list_tokenized_test = pad_sequences(list_tokenized_test, maxlen=max_length, truncating='post')

    pass


if __name__ == '__main__':
    main()