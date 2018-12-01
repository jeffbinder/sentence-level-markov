f = open('mobydick.chapters', 'r')

i = 0
chf = open('mobydick.chapterheadings', 'w')
for line in f.readlines():
    line = line.strip()
    if line.startswith('CHAPTER'):
        i += 1
        outf = open('mobydick/' + '{0:03}'.format(i), 'w')
        chf.write(line + '\n')
    else:
        outf.write(line + '\n')

