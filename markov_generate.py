# Generates text based on a sentence-level Markov model. This program will
# never repeat a sentence (unless multiple copies of the same sentence appear
# in the model). It starts with a random sentence; it will then pick the next
# sentence based on the model and repeat until it has exhausted all the
# sentences. There are two options for how to select sentences: it can either
# randomize the process using the weights as probabilities or it can simply
# pick the highest weight in all cases.

import os
import pickle
import re
import sys

import numpy

# Whether to include randomness.
RANDOMIZE = False

modelfname = sys.argv[1]
outfname = sys.argv[2]

coefs, sentences = pickle.load(open(modelfname, 'rb'))
nsentences = len(sentences)

# Some sentences will fit the decoder model better than others, and these
# will tend to have higher probabilities regardless of what sentence comes
# before. We normalize columns to account for this.
probs = numpy.zeros(coefs.shape)
for i in range(coefs.shape[1]):
    total = sum(coefs[:,i])
    if total != 0:
        probs[:,i] = coefs[:,i] / total

output = []
last_sentence = None

# Function to deal with a generated sentence.
def emit_sentence(sentnum):
    global last_sentence
    last_sentence = sentnum
    sent = sentences[sentnum]
    # Replace sentence-internal newlines with spaces so that the paragraphs
    # look right in the output.
    sent = re.sub(r'([^\s])\n+([^\s])', r'\1 \2', sent)
    # If the sentence begins with a single newline, replace it with a space.
    # This covers the case of sentences that happen to begin at the start of a
    # line but do not fall at the beginning of a paragraph.
    sent = re.sub(r'\A\n([^\n])', r' \1', sent)
    # If the sentence begins with more than two newlines, replace them with
    # two. This is to ensure that the paragraph formatting is consistent.
    sent = re.sub(r'\A\n\n\n+', r'\n\n', sent)
    output.append(sent)
    # Blank out the probability of generating this sentence since it has
    # already been used.
    probs[:,sentnum] = 0
    print(sentences[sentnum])
    
# Pick a sentence to start with.
if RANDOMIZE:
    emit_sentence(numpy.random.randint(0, nsentences))
else:
    # If we're not randomizing, just start with the first sentence.
    emit_sentence(0)

for i in range(nsentences - 1):
    p = probs[last_sentence,:]
    total = sum(p)
    if total == 0:
        # If we get here, there must be some sentences whose coefficients
        # in the model are all zero. Doing as the model says, we simply
        # do not include thiese sentences in our output.
        break
    p /= total
    if RANDOMIZE:
        emit_sentence(numpy.random.choice(nsentences, p=p))
    else:
        emit_sentence(numpy.argmax(p))

output = ' '.join(output)
f = open(outfname, 'w')
f.write(output)
