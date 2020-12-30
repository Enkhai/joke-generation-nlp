import numpy as np
from keras.models import load_model
import pickle
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize
from preprocessing import prepare_inputs


def generate(model, word2index, index2word, seed_text, next_words=100, method='beam_search'):
    if isinstance(model, str):
        model = load_model(model)
    if isinstance(word2index, str):
        word2index = pickle.load(open(word2index, 'rb'))
    if isinstance(index2word, str):
        index2word = pickle.load(open(index2word, 'rb'))

    time_steps = model.input_shape[0][1]
    if method == 'greedy':
        sentence = greedy_decode(model, word2index, index2word, seed_text, time_steps, next_words=next_words)
        sentence = ' '.join([index2word[token] for token in sentence])
    elif method == 'beam_search':
        sentences = beam_search_decode(model, word2index, seed_text, time_steps, next_words=next_words)
        sentence = ' '.join([index2word[token] for token in sentences[0][0]])

    return sentence


def greedy_decode(model, word2index, seed_text, time_steps, next_words):
    trim_suffix = True

    tokens = [word2index[word] for word in word_tokenize(seed_text.lower())]
    for _ in range(next_words):
        X, prev, _ = prepare_inputs([tokens[-(time_steps + 7):]], time_steps, len(word2index))
        pred = np.argmax(model.predict([X, prev]))

        if trim_suffix:
            tokens = tokens[:-1]
            trim_suffix = False
        tokens.append(pred)
    return tokens


def beam_search_decode(model, word2index, seed_text, times_steps, next_words, k=8):
    trim_suffix = True

    sequences = [[[word2index[word] for word in word_tokenize(seed_text.lower())], 0.0]]
    for _ in range(next_words):
        topk_indices = []
        topk_probas = []
        for seq in sequences:
            X, prev, _ = prepare_inputs([seq[0][-(times_steps + 7):]], times_steps, len(word2index))
            pred = model.predict([X, prev])[0]
            topk_idx = pred.argsort()[-k:][::-1]
            topk_prob = pred[topk_idx]

            topk_indices.append(topk_idx)
            topk_probas.append(topk_prob)

        if trim_suffix:
            sequences[0][0] = sequences[0][0][:-1]
            trim_suffix = False

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
    seed_text = 'The man in the white shirt took off his watch. "Why do you keep looking at me like that?", he asked.'
    # seed_text = 'Three children were playing in the park. One of them got up and looked at the other two.'
    # seed_text = "Once upon a time two guys were playing around with wires. Suddenly, one of them gets shocked."

    suffix = ' ' + 'the'
    print(generate(model, word2index, index2word, seed_text + suffix))
