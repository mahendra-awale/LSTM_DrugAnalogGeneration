import numpy as np
import os
import sys
from rdkit import Chem

# run the model on GPUs
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.layers import TimeDistributed

# ========================================================
# training set used for model building
training_set = sys.argv[1]

# model to use for sampling
modelwts_file = sys.argv[2]

# outputfile to store the data
output_file = sys.argv[3]

# below parameters are optional (4-7).
# default method for sampling:
# options are: a) numpychoice or b) multinomial
sampling_method = "numpychoice"

# default temp for multinomial sampling
temp_formultinomialsampling = 0.5

# default no of characters to sample
num_charstosample = 100000

# seed text for sampling
start_string = "C1CCCCC1"

if len(sys.argv) > 4:
    sampling_method = str(sys.argv[4])

if len(sys.argv) > 5:
    temp_formultinomialsampling = float(sys.argv[5])

if len(sys.argv) > 6:
    num_charstosample = int(sys.argv[6])

if len(sys.argv) > 7:
    start_string = str(sys.argv[7])

# ========================================================
print("your input:..")
print("trainid_set", training_set)
print("input_model", modelwts_file)
print("output_file", output_file)
print("sampling_method", sampling_method)
print("temp_formultinomialsampling", temp_formultinomialsampling)
print("num_charstosample", num_charstosample)
print("start_string", start_string)

text = open(training_set, 'r').read()
char_to_idx = {ch: i for (i, ch) in enumerate(sorted(list(set(text))))}
idx_to_char = {i: ch for (ch, i) in char_to_idx.items()}
vocab_size = len(char_to_idx)

print('Working on %d characters (%d unique)' % (len(text), vocab_size))
# ========================================================
SEQ_LEN = 1
BATCH_SIZE = 1
BATCH_CHARS = int(len(text) / BATCH_SIZE)
LSTM_SIZE = 512
LAYERS = 3


# ========================================================
def build_model():
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
test_model = build_model()
print('... done')


# ========================================================
def sample_multinomial(preds_probs, temperature=0.5):
    preds = np.asarray(preds_probs).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


# ========================================================
def sample_numpychoice(preds_probs, vocab_size):
    preds = np.asarray(preds_probs).astype('float64')
    preds = preds / np.sum(preds)
    sample = np.random.choice(range(vocab_size), p=preds)
    return sample


# ========================================================
def sample_compounds(model_wts_file, sample_chars=10000, primer_text='C1CCCCC1',
                     sample_method="numpychoice",
                     temprature="0.5"):
    test_model.reset_states()
    test_model.load_weights(model_wts_file)
    sampled = [char_to_idx[c] for c in primer_text]

    for c in primer_text:
        batch = np.zeros((1, 1, vocab_size))
        batch[0, 0, char_to_idx[c]] = 1
        test_model.predict_on_batch(batch)

    for i in range(sample_chars):
        batch = np.zeros((1, 1, vocab_size))
        batch[0, 0, sampled[-1]] = 1
        softmax = test_model.predict_on_batch(batch)[0].ravel()

        if sample_method == "numpychoice":
            sample = sample_numpychoice(softmax, vocab_size)
            sampled.append(sample)

        if sample_method == "multinomial":
            sample = sample_multinomial(softmax, temprature)
            sampled.append(sample)

    gen_smis = ''.join([idx_to_char[c] for c in sampled])
    return gen_smis.split("\n")


# ========================================================
def storevalidsmi(smis, outfilename):
    outfile = open(outfilename, "w")
    totalcpds = len(smis)
    validcpds = []
    for smi in smis:
        try:
            mol = Chem.MolFromSmiles(smi)
            smi = Chem.MolToSmiles(mol)
            if mol is None or smi == "" or smi == " ":
                continue
            else:
                validcpds.append(smi)
        except:
            continue

    validcpds_unq = {smi: "" for smi in validcpds}

    for smi in validcpds_unq:
        outfile.write(smi + " " + str(totalcpds) + "_" + str(len(validcpds)) + "_" + str(len(validcpds_unq)) + "\n")

    if len(validcpds_unq) == 0:
        outfile.write("C" + " " + str(totalcpds) + "_" + str(len(validcpds)) + "_" + str(len(validcpds_unq)) + "\n")

    outfile.close()


# ========================================================
smis_sampled = sample_compounds(modelwts_file, sample_chars=num_charstosample, primer_text=start_string,
                                sample_method=sampling_method,
                                temprature=temp_formultinomialsampling)

storevalidsmi(smis_sampled, output_file)
print("END")