# Splits a corpus into sentences so that it can be used as training data.

import os
import pickle
import sys

import nltk
from tensorflow import keras

indir = sys.argv[1]
outfname = sys.argv[2]

# Split all of the files into sentences and put them in a single list with
# '$' indicating file boundaries.
sentences = []
for fname in os.listdir(indir):
    f = open(os.path.join(indir, fname))
    text = f.read()
    text_sentences = nltk.sent_tokenize(text)
    sentences += [['$',]] + text_sentences
sentences += [['$',]]

# Convert the sentences to numerical data.
char_dict = {}
num_dict = {}
data = []
nchars = 0
for sentence in sentences:
    sentence_data = []
    for ch in sentence:
        if ch in char_dict:
            num = char_dict[ch]
            sentence_data.append(num)
        else:
            nchars += 1
            num = nchars
            char_dict[ch] = num
            num_dict[num] = ch
            sentence_data.append(num)
    data.append(sentence_data)

# Save the data.
outf = open(outfname, 'wb')
pickle.dump((char_dict, num_dict, data, sentences), outf)
