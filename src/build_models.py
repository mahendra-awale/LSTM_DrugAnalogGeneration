# Adapted from https://github.com/karpathy/char-rnn Implementation
import numpy as np
import os
import sys

# run the model on GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import TimeDistributed

# ========================================================
# input smiles file
training_set = sys.argv[1]

# output folder to store the models (wts are stored)
output_folder = sys.argv[2]

# number of epochs
num_epoch = int(sys.argv[3])

# number of samples (batch size) to feed to model for training at a time
batch_size = int(sys.argv[4])
# ========================================================

text = open(training_set, 'r').read()
char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(text))))}
idx_to_char = {i: ch for (ch, i) in char_to_idx.items()}
vocab_size = len(char_to_idx)

print('Working on %d characters (%d unique)' % (len(text), vocab_size))

# ========================================================
SEQ_LEN = 64
BATCH_SIZE = batch_size
BATCH_CHARS = int(len(text) / BATCH_SIZE)
LSTM_SIZE = 512
LAYERS = 3


# For training, each subsequent example for a given batch index should be a
# consecutive portion of the text.  To achieve this, each batch index operates
# over a disjoint section of the input text.
def read_batches(text):
    T = np.asarray([char_to_idx[c] for c in text], dtype=np.int32)
    X = np.zeros((BATCH_SIZE, SEQ_LEN, vocab_size))
    Y = np.zeros((BATCH_SIZE, SEQ_LEN, vocab_size))

    for i in range(0, BATCH_CHARS - SEQ_LEN - 1, SEQ_LEN):
        X[:] = 0
        Y[:] = 0
        for batch_idx in range(BATCH_SIZE):
            start = batch_idx * BATCH_CHARS + i
            for j in range(SEQ_LEN):
                X[batch_idx, j, T[start + j]] = 1
                Y[batch_idx, j, T[start + j + 1]] = 1

        yield X, Y


# ========================================================
def build_model(infer):
    if infer:
        batch_size = seq_len = 1
    else:
        batch_size = BATCH_SIZE
        seq_len = SEQ_LEN
    model = Sequential()
    model.add(LSTM(LSTM_SIZE,
                   return_sequences=True,
                   batch_input_shape=(batch_size, seq_len, vocab_size),
                   stateful=True))

    model.add(Dropout(0.2))
    for l in range(LAYERS - 1):
        model.add(LSTM(LSTM_SIZE, return_sequences=True, stateful=True))
        model.add(Dropout(0.2))

    model.add(TimeDistributed(Dense(vocab_size)))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adagrad')
    return model


# ========================================================
print('Building model.')
training_model = build_model(infer=False)
print("..done")

for epoch in range(num_epoch):
    for i, (x, y) in enumerate(read_batches(text)):
        loss = training_model.train_on_batch(x, y)
        print(epoch, i, loss)
    training_model.save_weights(output_folder + '/keras_char_rnn.%d.h5' % epoch,
                                overwrite=True)
print("END")
