#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''Example script to generate text from Nietzsche's writings.

At least 20 epochs are required before the generated text
starts sounding coherent.

It is recommended to run this script on GPU, as recurrent
networks are quite computationally intensive.

If you try this script on new data, make sure your corpus
has at least ~100k characters. ~1M is better.

--------------------------------------------------------------

Here are the possible four commands.

./lstm_text_generation alice type1
./lstm_text_generation alice type2
./lstm_text_generation obama type1
./lstm_text_generation obama type2

--------------------------------------------------------------

This script is a modified version of the original file found
in the following link.
https://github.com/fchollet/keras/blob/master/examples/lstm_text_generation.py
'''

from __future__ import print_function
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Dropout
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
import random
import sys
import codecs

args = sys.argv

input_data = args[1] # 'alice', 'obama'
model_type = args[2] # 'type1', 'type2'

if input_data == 'alice':
    input_path = '../input/alice.txt'
elif input_data == 'obama':
    input_path = '../input/obama.txt'

sys.stdin = codecs.getreader('utf_8')(sys.stdin)
sys.stdout = codecs.getwriter('utf_8')(sys.stdout)

#path = get_file('nietzsche.txt', origin="https://s3.amazonaws.com/text-datasets/nietzsche.txt")
#text = open(path).read().lower()
text = codecs.open(input_path, encoding='utf-8').read().lower()


print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

nb_iteration = 5
maxlen = 40 # cut the text in semi-redundant sequences of maxlen characters
step = 3
sentences = []
next_chars = []

for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))
print('nb iterations:', nb_iteration)

print('Vectorization...')
X = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        X[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

# build the model: a single LSTM
print('Build model...')

if model_type == 'type1':
    model = Sequential()
    model.add(LSTM(256, input_shape=(maxlen, len(chars))))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))
elif model_type == 'type2':
    model = Sequential()
    model.add(LSTM(256, return_sequences=True, input_shape=(maxlen, len(chars))))
    model.add(LSTM(256))
    model.add(Dense(len(chars)))
    model.add(Activation('softmax'))

optimizer = RMSprop(lr=0.01)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

acc = []
loss = []

# train the model, output generated text after each iteration

for iteration in range(1, nb_iteration):
    print()
    print('-' * 50)
    print('Iteration', iteration)
    history = model.fit(X, y, batch_size=128, nb_epoch=1)
    curr_acc = history.history['acc'][0]
    curr_loss = history.history['loss'][0]
    acc.append(curr_acc)
    loss.append(curr_loss)

    start_index = random.randint(0, len(text) - maxlen - 1)

    for diversity in [0.01, 0.1, 0.2, 0.5, 1.0, 1.2]:
        print()
        print('----- diversity:', diversity)

        generated = ''
        sentence = text[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                x[0, t, char_indices[char]] = 1.

            preds = model.predict(x, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]

            generated += next_char
            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

fig, ax = plt.subplots(2, 1, figsize=(10, 6))
ax[0].plot(acc, label='acc', color='b')
ax[0].set(ylabel='Accuracy', ylim=(0, 1))
ax[1].plot(loss, label='loss', color='r')
ax[1].set(ylabel='Loss')
figname = 'lstm_text_generation_{0}_{1}.png'.format(input_data, model_type)
fig.savefig(figname)

