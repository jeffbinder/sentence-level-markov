# A sentence-level Markov text generator

> "I am a Speculation of the Poet"

—a neural network

This is my entry for the 2018 version of [National Novel Generation Month](https://github.com/NaNoGenMo/2018), which challenges people to write a computer program that creates a 50,000-word novel.

Short version: I split each chapter of _Moby-Dick_ into sentences, then had an AI choose the order in which the sentences should appear. I call the result [_Mboy-Dcki_](https://raw.githubusercontent.com/jeffbinder/sentence-level-markov/master/mboydcki.txt).

Take, for instance, the opening of the famous chapter on "The Whiteness of the Whale." Here is Melville's version:

> WHAT the white whale was to Ahab, has been hinted; what, at times, he was to me, as yet remains unsaid.
> 
> Aside from those more obvious considerations touching Moby Dick, which could not but occasionally awaken in any man's soul some alarm, there was another thought, or rather vague, nameless horror concerning him, which at times by its intensity completely overpowered all the rest; and yet so mystical and well nigh ineffable was it, that I almost despair of putting it in a comprehensible form. It was the whiteness of the whale that above all things appalled me. But how can I hope to explain myself here; and yet, in some dim, random way, explain myself I must, else all these chapters might be naught.

Here is what my neural network put together:

> What the white whale was to Ahab, has been hinted; what, at times, he was to me, as yet remains unsaid. 
> 
> Bethink thee of the albatross, whence come those clouds of spiritual wonderment and pale dread, in which that white phantom sails in all imaginations?  Wonder ye then at the fiery hunt?  To analyse it, would seem impossible.  The flashing cascade of his mane, the curving comet of his tail, invested him with housings more resplendent than gold and silver-beaters could have furnished him.  He was the elected Xerxes of vast herds of wild horses, whose pastures in those days were only fenced by the Rocky Mountains and the Alleghanies.

## What I did

This project is inspired by a passage in Benjamin Franklin's _Autobiography_ in which he describes how he learned to write as a young man.

> About this time I met with an odd volume of the Spectator. It was the third. I had never before seen any of them. I bought it, read it over and over, and was much delighted with it. I thought the writing excellent, and wished, if possible, to imitate it. With this view I took some of the papers, and, making short hints of the sentiment in each sentence, laid them by a few days, and then, without looking at the book, try'd to compleat the papers again, by expressing each hinted sentiment at length, and as fully as it had been expressed before, in any suitable words that should come to hand. Then I compared my Spectator with the original, discovered some of my faults, and corrected them. But I found I wanted a stock of words, or a readiness in recollecting and using them, which I thought I should have acquired before that time if I had gone on making verses; since the continual occasion for words of the same import, but of different length, to suit the measure, or of different sound for the rhyme, would have laid me under a constant necessity of searching for variety, and also have tended to fix that variety in my mind, and make me master of it. Therefore I took some of the tales and turned them into verse; and, after a time, when I had pretty well forgotten the prose, turned them back again. I also sometimes jumbled my collections of hints into confusion, and after some weeks endeavored to reduce them into the best order, before I began to form the full sentences and compleat the paper. This was to teach me method in the arrangement of thoughts.

Using a Recurrent Neural Network (RNN), I created an AI that perfoms a bastardized version of this unscrambling process. Rather than working with hints of the ideas underlying the prose, this program works with sentences. First, it splits a text into sentences, and then it tries to figure out what order those sentences should go in based only on the sentences themselves.

I trained the neural net on 500,000 sentences (with 10% held out for evaluation) from the Wright American Fiction corpus, excluding _Moby-Dick_ itself. I then generated a new novel by having the AI decide the order of sentences within each chapter of _Moby-Dick_. [You can view the result here](https://raw.githubusercontent.com/jeffbinder/sentence-level-markov/master/mboydcki.txt).

Over all the chapters, the neural network is able to pick the right sentence ("right" meaning the one that appears next in the original text) about 3.5% of the time (328 / 9384 sentences chosen). (The success rate might be a little different for the actual text generator because of the constraint that each sentence only be used once.) This is not too impressive, but it is better than the expected success rate if the sentences were chosen randomly, which is 1.4%. Even when it makes a wrong decision, the model is not choosing sentences in a totally arbitrary way---it consistently ranks some sentences much higher than others.

Do these choices have any logic to them? I've yet to dig into it in detail, but I can offer some provisional observations. The network has learned to follow exclamation points with uncapitalized words, as was common in nineteenth-century novels; it also developed a rudimentary model of how quotation marks and dialogue tags fit together. It likes to follow one question with another. It also seems to have learned to group together sentences that contain similar pronouns (I/me, she/her, etc.). Whether it learned anything involving substantive words I do not know.

## The technical details

This program employs a variant of the [Markov chain method](https://en.wikipedia.org/wiki/Markov_chain), one of the oldest and simplest methods of computational text generation. The most common version of this method works by measuring which words appear most often after particular sequences of words; it then uses this information to predict what word will come next. Thus, if the last 6 words are "What are your plans for the," the model might predict that the next word will be "weekend." Markov chains are typically constructed at the character, word, or token level. However, it is also possible to create a Markov chain model that operates at the sentence level. For each sentence generated, the model must assign probabilities for every sentence that could potentially come next. It is generally not possible to train such a model directly, since each sentence typically only occurs once even in a very large amount of text. However, it is possible to generate a sentence-level Markov model by using a neural net to determine the probabilities. That is, in essence, what this software does.

The neural net I created takes in the last 25 characters of a sentence and tries to predict the first 25 characters of the next sentence. I used a sequential model, meaning that the network produces the output one character at a time, with the probabilities for each new character conditioned by what characters have already been generated. Instead of using the network to generate text directly, the software uses it to compute a probability for every pair of sentences in a given text. After a bit of normalization, these probabilities can be used in a Markov chain text generator. To maximize the uncanny effect, I set up the generator to start with the actual first sentence of the chapter. After that, it always picks the next sentence with the highest probability according to the Markov model, only using each sentence once. (There is an option to randomize the output as well.) I ran the generator separately for each chapter of _Moby-Dick_ because it has to run a neural network on every pair of sentences, which would take too long for the whole novel.

The network architecture I used is similar to one of the simplest types that has been used in [neural machine translation](https://en.wikipedia.org/wiki/Neural_machine_translation), an encoder-decoder network based on [LSTM (Long Short-Term Memory)](https://en.wikipedia.org/wiki/Long_short-term_memory) units. Instead of taking in a sentence and outputting a (supposed) translation, it takes in the last _n_ characters of a sentence and outputs the predicted first _n_ characters of the next sentence. As I explain in the code, I had to use a slightly different data format from the one used in machine translation to account for the fact that the input and output data do not represent complete linguistic units.

If you want to run the program yourself, you will need to install [Python 3](python.org), [NLTK](nltk.org), [TensorFlow](tensorglow.org), and [NumPy/SciPy](scipy.org). Then put your training data in a bunch of plain text files in a directory and the text you want to scramble in another plain text file. You can then do something like this at the command line:

```
python prepare_corpus.py <training-data-dir> <corpus-name>.corpus
python train.py <corpus-name>.corpus <corpus-name>.network
python markovize.py <corpus-name>.corpus <corpus-name>.network <input-text-name>.txt <output>.markov
python markov_generate.py <input>.markov <output>.txt
```

The scripts contain various variables that you can edit to alter the behavior of the model.

The training and Markovizing scripts may take hours or days to run on an ordinary computer, depending on the amount of data you are using.

In addition to the code, I included the trained neural network and Markov models I used in generating _Mboy-Dcki_. I also included the script generate_md.py, which automates running the process on every chapter of the novel and compute_success_rate.py, which I used to compute the statistics.

The text of _Moby-Dick_ is taken from Project Gutenberg.