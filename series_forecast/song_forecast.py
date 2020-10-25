import numpy as np
import scipy.io.wavfile as wfile
import matplotlib.pyplot as plt
import tensorflow as tf
import h5py


def read_files():
    org = wfile.read('org.wav')
    long = wfile.read('long2.wav')
    short = wfile.read('short2.wav')
    return org, long, short


def data_divider(x, y):
    split_time = 55000
    x_train = x[:split_time]
    y_train = y[:split_time]
    x_valid = x[split_time:]
    y_valid = y[split_time:]

    return x_train, y_train, x_valid, y_valid


def window_data(data, window_size, batch_size, shuffle_buffer):
    data = tf.expand_dims(data, axis=-1)
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.window(window_size + 1, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda d: d.batch(window_size + 1))
    ds = ds.shuffle(shuffle_buffer)
    ds = ds.map(lambda d: (d[:-1], d[1:]))

    return ds.batch(batch_size).prefetch(1)


def forecast(model, data, window_size):
    ds = tf.data.Dataset.from_tensor_slices(data)
    ds = ds.window(window_size, shift=1, drop_remainder=True)
    ds = ds.flat_map(lambda d: d.batch(window_size))
    ds = ds.batch(64).prefetch(1)
    prediction = model.predict(ds)

    return prediction


def train_model(org, window_size, batch_size, shuffle_size, normalized_param):
    y = np.asarray(org[1][10000:60000])
    x = np.asarray(org[1][9900:59999])
    y = np.divide(y, normalized_param)
    x = np.divide(x, normalized_param)
    x_train, y_train, x_valid, y_valid = data_divider(x, y)

    train_set = window_data(x_train, window_size, batch_size, shuffle_size)

    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv1D(filters=64, kernel_size=7, strides=1, padding="causal",
                               activation="relu", input_shape=[None, 1]),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.LSTM(64, return_sequences=True),
        tf.keras.layers.Dense(60, activation="relu"),
        tf.keras.layers.Dense(30, activation="relu"),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(1),
        tf.keras.layers.Lambda(lambda x: x * 1000)
    ])

    optimizer = tf.keras.optimizers.SGD(lr=1e-6, momentum=0.9)
    model.compile(loss=tf.keras.losses.Huber(),
                  optimizer=optimizer, metrics=["mae"])
    history = model.fit(train_set, epochs=100)


    ## uncomment below to adjust learning rate
    # lr_schedule = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-8 * 10 ** (epoch / 20))
    # history = model.fit(train_set, batch_size=256, epochs=50, callbacks=[lr_schedule])
    # moze = np.array(history.history["lr"])
    # pomoze = np.array(history.history["loss"])
    #
    # plt.semilogx(moze, pomoze)
    # plt.show()

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


def nn(org, long, short, model, window_size, normalized_param):
    auxi = np.array(org[1])
    auxi = auxi / normalized_param

    x = np.arange(50000, 60000, 1)
    y = np.asarray(org[1][50000:60000])

    short_copy = org[1].copy()

    short_prediction = forecast(model, auxi[..., np.newaxis], window_size)
    short_prediction = short_prediction[0:-1, -1, 0]
    short_prediction = np.multiply(short_prediction, normalized_param)

    fig, a = plt.subplots(1, 1)
    a.plot(short_prediction[54900:56400])
    a.plot(short_copy[55000:56500], color='red', lw=1)

    plt.show()


def main():
    window_size = 100
    batch_size = 128
    shuffle_size = 1000

    org, long, short = read_files()
    normalized_param = max(org[1].max(), abs(org[1].min()))
    train_model(org, window_size, batch_size, shuffle_size, normalized_param)
    # model = load_model()
    # nn(org[:60000], long[:60000], short[:60000], model, window_size, normalized_param)


if __name__ == '__main__':
    main()
