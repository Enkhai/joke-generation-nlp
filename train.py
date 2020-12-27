import pickle
import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dropout, Dense, Activation
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize
from generate import generate


def load_dataset(file):
    df = pd.read_csv(file)

    sentences = df.body.dropna().drop_duplicates()
    sentences_words = list(sentences.apply(lambda x: word_tokenize((str(x).lower()))))

    # sentences_words = [seq for seq in sentences_words if 200 < len(seq) < 350]

    word2index = {'PAD': 0}
    for seq in sentences_words:
        for word in seq:
            if word not in word2index:
                word2index[word] = len(word2index)
    index2word = {i: w for w, i in word2index.items()}

    input_sequences = []
    for sent in sentences_words:
        sent_tokens = [word2index[word] for word in sent]
        # for i in range(20, len(sent_tokens)):
        for i in range(1, len(sent_tokens)):
            input_sequences.append(sent_tokens[:i + 1])

    max_sequence_len = max([len(x) for x in input_sequences])
    input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

    X, Y = input_sequences[:, :-1], to_categorical(input_sequences[:, -1], len(word2index))

    return X, Y, word2index, index2word, max_sequence_len


if __name__ == '__main__':
    # X, Y, word2index, index2word, max_len = load_dataset('data/dataset.csv')
    X, Y, word2index, index2word, max_len = load_dataset('data/dataset1.csv')

    model = Sequential()
    model.add(Embedding(len(word2index), 100, input_length=max_len - 1))
    model.add(LSTM(500, return_sequences=True))
    model.add(Dropout(0.2))
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

    model.fit(X, Y, epochs=100, batch_size=128)

    pickle.dump(word2index, open('word2index.pickle', 'wb'))
    pickle.dump(index2word, open('index2word.pickle', 'wb'))
    model.save('model.h5')

    print(generate(model, word2index, index2word, 'The'))
