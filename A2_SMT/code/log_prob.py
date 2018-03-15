from preprocess import *
from lm_train import *
from math import log

def log_prob(sentence, LM, smoothing=False, delta=0, vocabSize=0):
    """
	Compute the LOG probability of a sentence, given a language model and whether or not to
	apply add-delta smoothing

	INPUTS:
	sentence :	(string) The PROCESSED sentence whose probability we wish to compute
	LM :		(dictionary) The LM structure (not the filename)
	smoothing : (boolean) True for add-delta smoothing, False for no smoothing
	delta : 	(float) smoothing parameter where 0<delta<=1
	vocabSize :	(int) the number of words in the vocabulary

	OUTPUT:
	log_prob :	(float) log probability of sentence
	"""
    word_list = sentence.split()

    if not smoothing:
        log_prob = calProb(word_list, LM, 0, vocabSize)
    else:
        log_prob = calProb(word_list, LM, delta, vocabSize)
    return log_prob


def calProb(word_list, LM, delta, vocabSize):
    sentence_len = len(word_list)
    logProb = 0

    for i in range(sentence_len - 1):
        if word_list[i] in LM['bi'] and word_list[i+1] in LM['bi'][word_list[i]]:
            countw1w2 = LM['bi'][word_list[i]][word_list[i+1]]
        else:
            countw1w2 = 0

        if word_list[i] in LM['uni']:
            countw1 = LM['uni'][word_list[i]]
        else:
            countw1 = 0

        # print(str(countw1w2) + " and " + str(countw1) + ' ' + word_list[i] + ' ' + word_list[i+1])
        if (countw1 == 0 or countw1w2 == 0) and delta == 0: # special case, return 0, or -inf in log space
            return float('-inf')
        logProb += log( (countw1w2 + delta) / (countw1 + delta*vocabSize), 2)

    return logProb