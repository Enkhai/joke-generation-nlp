import numpy as np
import pandas as pd
from keras.utils import to_categorical
import nltk

nltk.download('punkt')
from nltk.tokenize import word_tokenize


def load_dataset(file, time_steps=20):
    df = pd.read_csv(file)

    sentences = df.body.dropna().drop_duplicates()
    sentences_words = list(sentences.apply(lambda x: word_tokenize((str(x).lower()))))

    word_counts = {}
    for seq in sentences_words:
        for word in seq:
            try:
                word_counts[word] += 1
            except KeyError:
                word_counts[word] = 1

    rare_words = [word for word, count in word_counts.items() if count < 2]
    sentences_words = [seq for seq in sentences_words if
                       set(seq).isdisjoint(rare_words) and time_steps + 7 + 1 < len(seq)]

    word2index = {}
    index2word = {}
    for seq in sentences_words:
        for word in seq:
            if word not in word2index:
                i = len(word2index)
                word2index[word] = i
                index2word[i] = word

    sent_tokens = [[word2index[word] for word in sent] for sent in sentences_words]

    X, prev_one_hot, Y = prepare_inputs(sent_tokens, time_steps, len(word2index))

    return X, prev_one_hot, Y, word2index, index2word


def prepare_inputs(tokens, time_steps, vocab_size):
    input_sequences = np.array([sent[i - (time_steps + 7):i]
                                for sent in tokens
                                for i in range(time_steps + 7, len(sent) + 1)])

    X = np.array([[seq[i:i + 7] for i in range(time_steps)] for seq in input_sequences])
    prev = to_categorical(X[:, :, -1], vocab_size)
    Y = to_categorical(input_sequences[:, -1], vocab_size)

    return X, prev, Y
