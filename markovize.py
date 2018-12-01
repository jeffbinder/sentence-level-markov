# Generates a Markov model by applying the neural net to the sentences of
# a corpus (possibly the same one on which the network was trained).

import os
import pickle
import sys

import nltk
import numpy

import tensorflow as tf

# The model will include only the first NUM_SENTENCES sentences from the input
# data. Set to None to include all the data.
NUM_SENTENCES = None

corpusfname = sys.argv[1]
modelfname = sys.argv[2]
textfname = sys.argv[3]
outfname = sys.argv[4]

inf = open(corpusfname, 'rb')
char_dict, num_dict, _, _ = pickle.load(inf)

model = tf.keras.models.load_model(modelfname)

# Find out the dimensions of the data.
WINDOW_LEN = model.get_layer('enc_in').input_shape[1]
RNN_UNITS = model.get_layer('x1').units
nchars = len(char_dict) + 3
pad = 0
sent_start = nchars - 2
sent_end = nchars - 1

# Construct the encoder and decoder networks.
encoder_in = tf.keras.layers.Input(shape=(None, nchars),
                                   dtype='float32', name='enc_in')
encoder_out, enc_h, enc_c \
    = tf.keras.layers.LSTM(RNN_UNITS, return_state=True,
                           name='enc_out')(encoder_in)
encoder = tf.keras.models.Model(encoder_in, [enc_h, enc_c])

decoder_in = tf.keras.layers.Input(shape=(None, nchars),
                                   dtype='float32', name='dec_in')
decoder_in_h = tf.keras.layers.Input(shape=(RNN_UNITS,),
                                     name='dec_in_h')
decoder_in_c = tf.keras.layers.Input(shape=(RNN_UNITS,),
                                     name='dec_in_c')
x, decoder_h, decoder_c \
    = tf.keras.layers.LSTM(RNN_UNITS, return_sequences=True,
                           return_state=True,
                           name='x1')(decoder_in,
                                      initial_state=[decoder_in_h,
                                                     decoder_in_c])
output = tf.keras.layers.Dense(nchars, activation='softmax',
                               name='dec_out')(x)
decoder = tf.keras.models.Model([decoder_in_h, decoder_in_c, decoder_in],
                                [output, decoder_h, decoder_c])

# Copy the weights from the loaded model.
encoder.get_layer('enc_in').set_weights(model.get_layer('enc_in')
                                        .get_weights())
encoder.get_layer('enc_out').set_weights(model.get_layer('enc_out')
                                         .get_weights())
decoder.get_layer('dec_in').set_weights(model.get_layer('dec_in')
                                        .get_weights())
decoder.get_layer('x1').set_weights(model.get_layer('x1').get_weights())
decoder.get_layer('dec_out').set_weights(model.get_layer('dec_out')
                                         .get_weights())

# Read the text.
inf = open(textfname, 'r')
text = inf.read()

# Split into sentences.  We use span_tokenize() because we want to include
# whitespace preceding a sentence in the model so that we can predict
# paragraph breaks.
tknzr = nltk.tokenize.punkt.PunktSentenceTokenizer()
spans = tknzr.span_tokenize(text)
sentences = [text[0:spans[0][1]]]
last_end = spans[0][1]
for start, end in spans[1:]:
    sentences.append(text[last_end:end])
    last_end = end
if NUM_SENTENCES:
    sentences = sentences[:NUM_SENTENCES]
nsentences = len(sentences)

# Convert the sentences to numerical data to be used as input for the model.
data = numpy.zeros((nsentences, WINDOW_LEN, nchars))
for i in range(nsentences):
    sentence = sentences[i].strip()
    s1 = [char_dict.get(c, 0) for c in sentence]
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
        data[i, j + s1_padding, idx] = 1

# Functions for dealing with the model.
def compute_sentence_probability(h, c, s2):
    prob = 1.0
    last_char = sent_start
    slen = len(s2)
    for i in range(min(slen + 1, WINDOW_LEN)):
        if i == slen:
            char = sent_end
        else:
            char = char_dict.get(s2[i], 0)
        datum = numpy.zeros((1, 1, nchars))
        datum[0, 0, last_char] = 1
        x, h, c = decoder.predict([h, c, datum])
        p = x[0, 0, char]
        prob *= p
        last_char = char
    return prob
def gen_next_sentence(h, c):
    s2 = []
    last_char = sent_start
    for i in range(WINDOW_LEN):
        datum = numpy.zeros((1, 1, nchars))
        datum[0, 0, last_char] = 1
        x, h, c = decoder.predict([h, c, datum])
        char = numpy.argmax(x[0, :])
        if char == sent_end:
            break
        s2.append(num_dict.get(char, ''))
        last_char = char
    return ''.join(s2)
            
# Compute the probability for each pair of sentences.
probs = numpy.zeros((nsentences, nsentences))
print("Generating juxtaposition matrix...")
for i in range(nsentences):
    s1 = data[i]
    s1 = numpy.array([s1])
    h, c = encoder.predict(s1)
    for j in range(nsentences):
        s2 = sentences[j].strip()
        p = compute_sentence_probability(h, c, s2)
        probs[i,j] = p
    if i % 10 == 0:
        print('\n----------------')
        print(str(i) + ' / ' + str(nsentences))
        print('Sentence: ' + sentences[i])
        print('')
        print('Generated next sentence: '
              + gen_next_sentence(h, c))
        print('')
        if i < nsentences - 1:
            print('Actual next sentence: '
                  + sentences[i+1])
        else:
            print('Actual next sentence: <EOD>')

# Save the Markov model.
outf = open(outfname, 'wb')
pickle.dump((probs, sentences), outf)

