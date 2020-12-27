import pickle
import numpy as np
import pandas as pd
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, Activation
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
from generate import generate


def load_dataset(file):
    df = pd.read_csv(file)

    sequences = df.body.dropna().drop_duplicates()
    sequences = list(sequences.apply(lambda x: word_tokenize((str(x).lower()))))

    word_counts = {}
    for seq in sequences:
        for word in seq:
            try:
                word_counts[word] += 1
            except KeyError:
                word_counts[word] = 1

    rare_words = [word for word, count in word_counts.items() if count < 3]
    sequences = [seq for seq in sequences if set(seq).isdisjoint(rare_words) and len(seq) <= 250 - 2]

    word2index = {'SOS': 0,
                  'EOS': 1}
    for seq in sequences:
        for word in seq:
            if word not in word2index:
                word2index[word] = len(word2index)

    index2word = {i: w for w, i in word2index.items()}

    sequences = [[0] + [word2index[word] for word in seq] + [1] for seq in sequences]

    X = [np.array(seq[:-1]) for seq in sequences]
    Y = [to_categorical(seq[1:], len(index2word)) for seq in sequences]

    return X, Y, word2index, index2word


if __name__ == '__main__':
    X, Y, word2index, index2word = load_dataset('data/dataset.csv')

    model = Sequential()
    model.add(Embedding(len(word2index), 100, input_length=1))
    model.add(LSTM(500, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(500, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(500))
    model.add(Dropout(0.2))
    model.add(Dense(len(word2index)))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy')
    model.summary()

    for epoch in range(3):
        for step, (x, y) in enumerate(zip(X, Y)):
            model.fit(x, y, batch_size=250, shuffle=False, verbose=0)
            # model.reset_states()
            if step % 100 == 0:
                print('Epoch: {}, step: {}'.format(epoch, step))

    pickle.dump(word2index, open('word2index.pickle', 'wb'))
    pickle.dump(index2word, open('index2word.pickle', 'wb'))
    model.save('model.h5')

    print(generate(model, word2index, index2word, method='greedy', prefix='The'))
