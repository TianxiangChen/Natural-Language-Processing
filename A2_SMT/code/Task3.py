from preprocess import *
from lm_train import *
from log_prob import *
from perplexity import *
import numpy as np
import pickle


def generateLM(filename, language, train_dir):
    fullname = filename + '.pickle'
    if os.path.isfile(fullname):
        with (open(fullname, "rb")) as openfile:
            LM = pickle.load(openfile)
    else:
        LM = lm_train(train_dir, language, filename)
    return LM

if __name__ == "__main__":

	train_dir = '../data/Hansard/Training/'
	test_dir = '../data/Hansard/Testing/'

	LM_e = generateLM('LM_e', 'e', train_dir)
	LM_f = generateLM('LM_f', 'f', train_dir)
	preplexity_e = []
	preplexity_f = []

	for i in np.arange(0,1,0.2):
		preplexity_e.append(preplexity(LM_e, test_dir, 'e',True, i))
		preplexity_f.append(preplexity(LM_f, test_dir, 'f',True, i))

	sent1 = "Applying smoothing delta from 0 to 1, step by 0.2, the result is:"
	sent2 = "For Engilish, the Perplexity are: {}".format(preplexity_e)
	sent3 = "For French, the Perplexity are: {}".format(preplexity_f)
	sent4 = 'Perplexity is a measurement of how well a probability distribution or probability model predicts a sample.'
	sent5 = 'When applying smoothing, it puts some probability to the unseen data, while lowering the probability for the seen ones.'
	sent6 = "Thus, as the delta goes up (put more prob for unseen data), while the test sample isn't that complicated (contain lots of unseen data from training), the perlexity goes down, which is the case here."
	print(sent1 + '\n' + sent2 + '\n' + sent3 + '\n' + sent5 + '\n' + sent6)

	if os.path.isfile('Task3.txt'):
		os.remove('Task3.txt')
	with open('Task3.txt', 'a') as f:
		f.write(sent1 + '\n' + sent2 + '\n' + sent3 + '\n\n' + sent5 + '\n' + sent6)
