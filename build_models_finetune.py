# Adapted from https://github.com/karpathy/char-rnn Implementation
import numpy as np
import os
import sys

# run thr model on GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import TimeDistributed
from keras import optimizers

# ========================================================
# training set used to trained the model
training_set_original = sys.argv[1]

# model to fine tune (this model was built using sys.argv[1])
model_wts_tostart = sys.argv[2]

# new training set on which model will be finetune
training_set_fortuning = sys.argv[3]

# number epochs
num_epoch = int(sys.argv[4])

# batch size
batch_size = int(sys.argv[5])

# learning rate
learning_rate = float(sys.argv[6])

# output folder to store the fine tune models
output_folder = sys.argv[7]

# ========================================================
print("input:")
print("training_set_original", training_set_original)
print("model_wts_tostart", model_wts_tostart)
print("training_set_fortuning", training_set_fortuning)
print("num_epoch", num_epoch)
print("batch_size", batch_size)
print("learning_rate", learning_rate)
print("output_folder", output_folder)
# ========================================================

text = open(training_set_original, 'r').read()
char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(text))))}
idx_to_char = {i: ch for (ch, i) in char_to_idx.items()}
vocab_size = len(char_to_idx)

text = open(training_set_fortuning, 'r').read()

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

    adagrad = optimizers.Adagrad(lr=learning_rate)
    model.compile(loss='categorical_crossentropy', optimizer=adagrad)
    return model


# ========================================================
print('Building model.')
training_model = build_model(infer=False)
training_model.load_weights(model_wts_tostart)
print("..done")

for epoch in range(num_epoch):
    for i, (x, y) in enumerate(read_batches(text)):
        loss = training_model.train_on_batch(x, y)
        print(epoch, i, loss)
    training_model.save_weights(output_folder + '/keras_char_rnn.%d.h5' % epoch,
                                overwrite=True)
print("END")
