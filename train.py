import pickle
from keras.models import Sequential, Model
from keras.layers import Input, Embedding, Conv1D, LSTM, Dense, TimeDistributed, Activation
from keras.backend import squeeze
from generate import generate
from preprocessing import load_dataset
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


def create_model(time_steps, vocab_size, q):
    input_a = Input((time_steps, 7))
    embedding = Embedding(vocab_size, q, input_length=time_steps)(input_a)
    cnn = Sequential()
    cnn.add(Conv1D(q, 2, name='conv1_2'))
    cnn.add(Conv1D(q, 2, name='conv2_2'))
    cnn.add(Conv1D(q, 3, name='conv3_3'))
    cnn.add(Conv1D(q, 3, name='conv4_3'))
    cnn = squeeze(cnn(embedding), 2)
    rcm = LSTM(q, return_sequences=True)(cnn)

    input_b = Input((time_steps, vocab_size))
    prev_token_dense = TimeDistributed(Dense(q))(input_b)
    rgm = LSTM(q)(rcm + prev_token_dense)

    output = Activation('softmax')(Dense(vocab_size)(rgm))

    return Model(inputs=[input_a, input_b], outputs=output)


if __name__ == '__main__':
    steps = 20
    X, prev, Y, word2index, index2word = load_dataset('data/dataset.csv', time_steps=steps)

    model = create_model(steps, len(word2index), 200)

    model.compile(loss='categorical_crossentropy')
    model.summary()

    model.fit([X, prev], Y, epochs=20)

    pickle.dump(word2index, open('word2index.pickle', 'wb'))
    pickle.dump(index2word, open('index2word.pickle', 'wb'))
    model.save('model.h5')

    seed_text = 'The man in the white shirt took off his watch. "Why do you keep looking at me like that?", he asked.'
    suffix = ' ' + 'the'
    print(generate(model, word2index, index2word, seed_text + ' the'))
