import numpy as np
from keras.models import load_model
from keras.models import Sequential
from keras.utils import to_categorical
import pickle

from vocab import Vocabulary


def generate(model, vocab, prefix='A'):
    if isinstance(model, str):
        model: Sequential = load_model(model)
    if isinstance(vocab, str):
        vocab: Vocabulary = pickle.load(open(vocab, 'rb'))

    sentence = [1, vocab.word2index(prefix)]
    for token in sentence:
        x = to_categorical(token, vocab.num_words)
        pred = np.argmax(model.predict(x))
    sentence.append(pred)

    while True:
        x = to_categorical(pred, vocab.num_words)
        pred = np.argmax(model.predict(x))
        sentence.append(pred)
        if pred == 2:
            break

    return ' '.join([vocab.index2word(token) for token in sentence])


if __name__ == '__main__':
    model = '3_LSTM_500.h5'
    vocab = 'vocab.pickle'
    print(generate(model, vocab))
