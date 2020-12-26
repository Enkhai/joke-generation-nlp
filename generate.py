import numpy as np
from keras.models import load_model
import pickle


def generate(model, word2index, index2word, prefix='A'):
    if isinstance(model, str):
        model = load_model(model)
    if isinstance(word2index, str):
        word2index = pickle.load(open(word2index, 'rb'))
    if isinstance(index2word, str):
        index2word = pickle.load(open(index2word, 'rb'))

    sentence = [word2index[prefix.lower()]]
    for token in [0] + sentence:
        pred = np.argmax(model.predict([token]))
    sentence.append(pred)

    while pred != 1:
        pred = np.argmax(model.predict([pred.item()]))
        sentence.append(pred)

    return ' '.join([index2word[token] for token in sentence[:-1]])


if __name__ == '__main__':
    model = 'model.h5'
    word2index = 'word2index.pickle'
    index2word = 'index2word.pickle'
    print(generate(model, word2index, index2word))
