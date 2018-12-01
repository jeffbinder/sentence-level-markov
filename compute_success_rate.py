import math
import numpy
import os
import scipy.stats
import pickle
import matplotlib.pyplot as plt

ncorrect_overall = 0
nsentences_overall = 0
npairs_overall = 0
expected_overall = 0

for fname in sorted(os.listdir('mobydick-markovs')):
    coefs, sentences = pickle.load(open(os.path.join("mobydick-markovs",
                                                     fname),
                                        'rb'))
    nsentences = len(sentences)
    normalized = numpy.zeros(coefs.shape)
    for i in range(coefs.shape[1]):
        normalized[:,i] = coefs[:,i] / sum(coefs[:,i])

    correct_predictions = []
    npairs = nsentences - 1
    for i in range(npairs):
        prediction = numpy.argmax(normalized[i,:])
        correct_predictions.append(prediction == i + 1)
    ncorrect = len(list(filter(lambda x: x, correct_predictions)))

    ncorrect_overall += ncorrect
    nsentences_overall += nsentences
    npairs_overall += npairs
    expected_overall += (1.0 / nsentences) * npairs

    print(fname)
    print("Success rate: ", ncorrect / npairs,
          " (", ncorrect, " / ", npairs, ")", sep="")
    print("Expected success rate with random selection: ",
          1.0 / nsentences)
    print("----------")
    
print("Overall success rate ", ncorrect_overall / npairs_overall,
      " (", ncorrect_overall, " / ", npairs_overall, ")", sep="")
print("Expected success rate with random selection: ",
      expected_overall / npairs_overall,
      " (", expected_overall, " / ", npairs_overall, ")", sep="")
