import numpy as np
from keras.models import load_model
import pickle
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize


def generate(model, word2index, index2word, seed_text, next_words=100, method='beam_search'):
    if isinstance(model, str):
        model = load_model(model)
    if isinstance(word2index, str):
        word2index = pickle.load(open(word2index, 'rb'))
    if isinstance(index2word, str):
        index2word = pickle.load(open(index2word, 'rb'))

    max_len = model.input_shape[1]
    if method == 'greedy':
        sentence = greedy_decode(model, word2index, index2word, seed_text, max_len, next_words=next_words)
    elif method == 'beam_search':
        sentences = beam_search_decode(model, word2index, seed_text, max_len, next_words=next_words)
        sentence = ' '.join([index2word[token] for token in sentences[0][0]])

    return sentence


def greedy_decode(model, word2index, index2word, seed_text, max_sequence_length, next_words):
    for _ in range(next_words):
        tokens = [word2index[word] for word in word_tokenize((seed_text.lower()))]
        pred = np.argmax(model.predict([tokens[-max_sequence_length:]]))

        seed_text += ' ' + index2word[pred]
    return seed_text


def beam_search_decode(model, word2index, seed_text, max_sequence_length, next_words, k=8):
    sequences = [[[word2index[word] for word in word_tokenize(seed_text.lower())], 0.0]]
    for _ in range(next_words):
        topk_indices = []
        topk_probas = []
        for seq in sequences:
            pred = model.predict([seq[0][-max_sequence_length:]])[0]
            topk_idx = pred.argsort()[-k:][::-1]
            topk_prob = pred[topk_idx]

            topk_indices.append(topk_idx)
            topk_probas.append(topk_prob)

        all_candidates = []
        for (tokens, score), topk_idx, topk_prob in zip(sequences, topk_indices, topk_probas):
            for idx, prob in zip(topk_idx, topk_prob):
                all_candidates.append([tokens + [int(idx)], score - np.log(prob)])
        sequences = sorted(all_candidates, key=lambda tup: tup[1])[:k]
    return sequences


if __name__ == '__main__':
    model = 'model.h5'
    word2index = 'word2index.pickle'
    index2word = 'index2word.pickle'

    # ignore input incompatible shape
    # seed_text = 'The man in the white suit tipped his hat. "Why do you keep looking at me like that?", he asked.'
    # seed_text = 'Three children were playing in the park. One of them got up and looked at the other two.'
    seed_text = "Once upon a time two guys were playing around with wires. Suddenly, one of them gets shocked."
    print(generate(model, word2index, index2word, seed_text))
