# Reconstructs every chapter of Moby-Dick.

import os

# This is a lazy way to do it by w/e.
for fname in sorted(os.listdir('mobydick')):
    print(fname)
    if os.system('python markovize.py wright-nomoby.corpus' +
                 ' wright256-25-500000.network' +
                 ' mobydick/' + fname +
                 ' mobydick-markovs/' + fname + '.markov'):
        exit()
    if os.system('python markov_generate.py' +
                 ' mobydick-markovs/' + fname + '.markov' +
                 ' mboydcki/' + fname):
        exit()

outf = open('mboydcki.txt', 'w')
chf = open('mobydick.chapterheadings', 'r')
i = 0
for heading in chf.readlines():
    i += 1
    inf = open('mboydcki/' + '{0:03}'.format(i), 'r')
    outf.write(heading)
    outf.write(inf.read())
    outf.write('\n\n\n\n')
