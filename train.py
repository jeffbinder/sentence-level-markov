# Trains the model based on a sequence of sentences.

import gc
import sys
import pickle

import numpy
import tensorflow as tf

# The model will look at the last WINDOW_LEN chars of the first sentence and
# the first WINDOW_LEN chars of the second.
WINDOW_LEN = 25
# Training will include only the first NUM_SENTENCES sentences from the input
# data. Set to None to include all the data.
NUM_SENTENCES = 500000

# Number of units in the encoder/decoder networks. It's generally best to use
# smaller networks than one would typically use in machine translation because
# this model has a tendency to overfit.
RNN_UNITS = 256
# Number of epochs in the training process.
NUM_EPOCHS = 20
# Dropout rate for the encoder and decoder RNNs.
ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2

# Use this option if you want to resume training a model that was already
# saved.
RETRAIN = False

infname = sys.argv[1]
outfname = sys.argv[2]

# Load the processed text data.
inf = open(infname, 'rb')
char_dict, num_dict, sentences, _ = pickle.load(inf)

if NUM_SENTENCES:
    sentences = sentences[:NUM_SENTENCES]
nsentences = len(sentences)

# Add special elements to the vocabulary.
nchars = len(char_dict) + 3
pad = 0
sent_start = nchars - 2
sent_end = nchars - 1

# Prepare the training data.
training_enc_in = numpy.zeros((nsentences, WINDOW_LEN, nchars))
training_dec_in = numpy.zeros((nsentences, WINDOW_LEN, nchars))
training_output = numpy.zeros((nsentences, WINDOW_LEN, nchars))
for i in range(nsentences - 1):
    # Prepare the encoder input: the last n characters of sentence i. If the
    # sentence has fewer than i characters, it is padded on the left.
    s1 = sentences[i]
    s1len = len(s1)
    if s1len < WINDOW_LEN:
        s1_padding = WINDOW_LEN - s1len
        s1_start = 0
    else:
        s1_padding = 0
        s1_start = s1len - WINDOW_LEN
    s1_end = s1len
    for j in range(s1_end - s1_start):
        idx = s1[j + s1_start]
        training_enc_in[i, j + s1_padding, idx] = 1
    # Prepare the decoder input and output: the first n characters of sentence
    # i+1. The input is shifted one character to the right with a start
    # token at the beginning, as in seq2seq translation models. If the sentence
    # has fewer than i characters, a special end token appears at the end
    # of the output. Unlike in translation models, the end token is not
    # included in the output if the data reach all the way to the end of the
    # window. This is because we are not trying to produce an entire sentence,
    # but rather just to predict the first WINDOW_LEN characters of the
    # sentence; the end token would thus sometimes appear in the middle of a
    # word, messing up the model's ability to learn vocabulary.
    s2 = sentences[i + 1]
    s2len = len(s2)
    s2_end = min(s2len, WINDOW_LEN)
    training_dec_in[i, 0, sent_start] = 1
    for j in range(s2_end):
        idx = s2[j]
        if j < s2_end - 1:
            training_dec_in[i, j + 1, idx] = 1
        training_output[i, j, idx] = 1
    if s2_end < WINDOW_LEN:
        training_output[i, s2_end, sent_end] = 1

if RETRAIN:
    model = tf.keras.models.load_model(outfname)
    
else:
    # Create the neural network using the functional API.
    encoder_in = tf.keras.layers.Input(shape=(WINDOW_LEN, nchars),
                                       dtype='float32', name='enc_in')
    encoder_out, enc_h, enc_c \
        = tf.keras.layers.LSTM(RNN_UNITS, return_state=True,
                               dropout=ENC_DROPOUT, 
                               recurrent_activation='sigmoid', 
                               recurrent_initializer='glorot_uniform',
                               name='enc_out')(encoder_in)
    
    decoder_in = tf.keras.layers.Input(shape=(WINDOW_LEN, nchars),
                                       dtype='float32', name='dec_in')
    x = tf.keras.layers.LSTM(RNN_UNITS, return_sequences=True,
                             dropout=DEC_DROPOUT, 
                             recurrent_activation='sigmoid', 
                             recurrent_initializer='glorot_uniform',
                             name='x1')(decoder_in,
                                        initial_state=[enc_h, enc_c])
    output = tf.keras.layers.Dense(nchars, activation='softmax',
                                   name='dec_out')(x)

    # Compile the model.
    model = tf.keras.models.Model(inputs=[encoder_in, decoder_in],
                                  outputs=output)
    model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=0.001),
                  loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])

# Train and evaluate the model.
tf.logging.set_verbosity(tf.logging.ERROR)
checkpoint = tf.keras.callbacks.ModelCheckpoint(
    outfname + '.{epoch:02d}-{val_categorical_accuracy:.2f}',
    monitor='val_acc')
model.fit([training_enc_in, training_dec_in], training_output,
          batch_size=32, epochs=NUM_EPOCHS, shuffle=True,
          validation_split=0.1, callbacks=[checkpoint])

# Save the model.
tf.keras.models.save_model(model, outfname)
