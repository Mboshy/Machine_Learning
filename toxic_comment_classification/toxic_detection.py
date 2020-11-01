import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


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


def save_model(model):
    model_json = model.to_json()
    with open("model.json", "w") as json_file:
        json_file.write(model_json)

    model.save_weights("model.h5")
    print("Saved model to disk")


def load_model():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = tf.keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")

    return loaded_model


def main():
    max_length = 120
    oov_tok = "<OOV>"
    max_words = 20000
    embedding_dim = 16

    train, test = load_csv()

    classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
    train_labels = train[classes].values
    list_sentences_train = train["comment_text"]

    test_labels = pd.read_csv('test_labels.csv')
    list_sentences_test = pd.merge(test, test_labels, on='id')
    list_sentences_test = list_sentences_test[list_sentences_test['toxic'] != -1]
    test_labels = list_sentences_test[classes].values
    list_sentences_test = list_sentences_test["comment_text"]

    tokenizer = Tokenizer(num_words=max_words, oov_token=oov_tok)
    tokenizer.fit_on_texts(list(list_sentences_train))
    list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
    list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)

    list_tokenized_train = pad_sequences(list_tokenized_train, maxlen=max_length, truncating='post')
    list_tokenized_test = pad_sequences(list_tokenized_test, maxlen=max_length, truncating='post')

    model = tf.keras.Sequential([
        tf.keras.layers.Embedding(max_words, embedding_dim, input_length=max_length),
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(6, activation='sigmoid')
    ])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()

    num_epochs = 2
    history = model.fit(list_tokenized_train, train_labels, epochs=num_epochs, validation_split=0.05)

    print("Evaluation results on test set:")
    model.evaluate(list_tokenized_test, test_labels)

    save_model(model)

    plot_graphs(history, "accuracy")
    plot_graphs(history, "loss")


if __name__ == '__main__':
    main()
