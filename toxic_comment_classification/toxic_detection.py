# import pandas as pd
# import numpy as np
#
# df = pd.read_csv("train.csv")
# df['comment_text'] = df['comment_text'].str.lower()
# pass

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
# from keras.layers import Dense, LSTM, Embedding, Dropout, Activation
# from keras.layers import Bidirectional, GlobalMaxPool1D
# from keras.models import Model
# from keras import initializers, regularizers, constraints, optimizers, layers


def load_csv():
    train = pd.read_csv('train.csv')
    test = pd.read_csv('test.csv')
    return train, test


def plot_graphs(history, string):
    plt.plot(history.history[string])
    plt.plot(history.history['val_' + string])
    plt.xlabel("Epochs")
    plt.ylabel(string)
    plt.legend([string, 'val_' + string])
    plt.show()


# plot_graphs(history, "accuracy")
# plot_graphs(history, "loss")


def main():
    max_length = 120
    oov_tok = "<OOV>"
    max_words = 20000
    embedding_dim = 64

    train, test = load_csv()

    classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    train_labels = train[classes].values
    test_labels = pd.read_csv('test_labels.csv')[classes].values
    list_sentences_train = train["comment_text"]
    list_sentences_test = test["comment_text"]

    tokenizer = Tokenizer(num_words=max_words, oov_token=oov_tok)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

    list_tokenized_train = pad_sequences(list_tokenized_train, maxlen=max_length, truncating='post')
    list_tokenized_test = pad_sequences(list_tokenized_test, maxlen=max_length, truncating='post')

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(max_words, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(64, return_sequences=True)),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    num_epochs = 3
    history = model.fit(list_tokenized_train, train_labels, epochs=num_epochs, validation_data=(list_tokenized_test, test_labels))

    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")


if __name__ == '__main__':
    main()