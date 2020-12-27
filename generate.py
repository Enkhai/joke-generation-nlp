import numpy as np
from keras.models import load_model
import pickle


def generate(model, word2index, index2word, method='beam_search', prefix='A'):
    if isinstance(model, str):
        model = load_model(model)
    if isinstance(word2index, str):
        word2index = pickle.load(open(word2index, 'rb'))
    if isinstance(index2word, str):
        index2word = pickle.load(open(index2word, 'rb'))

    prefix = word2index[prefix.lower()]
    if method == 'greedy':
        sentence = greedy_decode(model, prefix)
    elif method == 'beam_search':
        sequences = beam_search_decode(model, prefix)
        sentence = sequences[0][0]

    return ' '.join([index2word[token] for token in sentence])


def greedy_decode(model, prefix):
    sentence = [prefix]
    for token in [0] + sentence:
        pred = np.argmax(model.predict([token]))
    sentence.append(pred)

    while pred != 1:
        pred = np.argmax(model.predict([pred.item()]))
        sentence.append(pred)

    return sentence[:-1]


def beam_search_decode(model, prefix, k=5):
    collected = []

    sequences = [[[prefix], 0.0]]
    while True:
        topk_indices = []
        topk_probas = []
        for seq in sequences:
            pred = model.predict([seq[0][-1]])[0]
            topk_idx = pred.argsort()[-k:][::-1]
            topk_prob = pred[topk_idx]

            topk_indices.append(topk_idx)
            topk_probas.append(topk_prob)

        all_candidates = []
        for (tokens, score), topk_idx, topk_prob in zip(sequences, topk_indices, topk_probas):
            for idx, prob in zip(topk_idx, topk_prob):
                if idx == 1:
                    collected.append([tokens, score])
                    k -= 1
                    if k == 0:
                        return sorted(collected, key=lambda tup: tup[1])
                else:
                    all_candidates.append([tokens + [int(idx)], score - np.log(prob)])
        sequences = sorted(all_candidates, key=lambda tup: tup[1])[:k]


if __name__ == '__main__':
    model = 'model.h5'
    word2index = 'word2index.pickle'
    index2word = 'index2word.pickle'
    print(generate(model, word2index, index2word, prefix='i'))
