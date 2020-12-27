import numpy as np
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import pickle
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize


def generate(model, word2index, index2word, seed_text, next_words=20, method='beam_search'):
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
        tokens = pad_sequences([tokens], maxlen=max_sequence_length, padding='pre')
        pred = np.argmax(model.predict(tokens))

        seed_text += ' ' + index2word[pred]
    return seed_text


def beam_search_decode(model, word2index, seed_text, max_sequence_length, next_words, k=8):
    sequences = [[[word2index[word] for word in word_tokenize(seed_text.lower())], 0.0]]
    for _ in range(next_words):
        topk_indices = []
        topk_probas = []
        for seq in sequences:
            tokens = pad_sequences([seq[0]], maxlen=max_sequence_length, padding='pre')
            pred = model.predict(tokens)[0]
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
    print(generate(model, word2index, index2word, 'i am'))
