import pickle
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, Activation
import pandas as pd
from generate import generate
from vocab import Vocabulary
import nltk

nltk.download([])


def load_dataset(file=''):
    pd.read_csv(file)
    x = []
    y = []
    vocab = Vocabulary()
    return x, y, vocab


def make_model(vocab_size, emb_size=100, sequence_length=20):
    model = Sequential()
    model.add(Embedding(vocab_size, emb_size, input_length=sequence_length))
    model.add(LSTM(500, return_sequences=True))
    model.add(LSTM(500, dropout=0.2, return_sequences=True))
    model.add(LSTM(500, dropout=0.2))
    model.add(Dropout(0.2))
    model.add(Dense(vocab_size))
    model.add(Activation('softmax'))

    return model


if __name__ == '__main__':
    # Part 2: Data preparation
    x, y, vocab = load_dataset()
    pickle.dump(vocab, open('vocab.pickle', 'wb'))

    # Part 3: Modelling
    model = make_model(100)
    model.compile(loss='categorical_crossentropy')
    model.summary()

    # Part 4: Training
    model.fit(x, y, epochs=5, shuffle=False)
    model.save('3_LSTM_500.h5')

    # Part 5: Evaluation
    print(generate(model, vocab))
