from preprocess import *
import pickle
import os

def lm_train(data_dir, language, fn_LM):
    """
	This function reads data from data_dir, computes unigram and bigram counts,
	and writes the result to fn_LM

	INPUTS:

    data_dir	: (string) The top-level directory continaing the data from which
					to train or decode. e.g., '/u/cs401/A2_SMT/data/Toy/'
	language	: (string) either 'e' (English) or 'f' (French)
	fn_LM		: (string) the location to save the language model once trained

    OUTPUT

	LM			: (dictionary) a specialized language model

	The file fn_LM must contain the data structured called "LM", which is a dictionary
	having two fields: 'uni' and 'bi', each of which holds sub-structures which
	incorporate unigram or bigram counts

	e.g., LM['uni']['word'] = 5 		# The word 'word' appears 5 times
		  LM['bi']['word']['bird'] = 2 	# The bigram 'word bird' appears 2 times.
    """

    LM = {}
    LM['uni'] = {}
    LM['bi'] = {}

    for root, dirs, files in os.walk(data_dir, topdown=True):
        files.sort()
        for name in files:
            if name[-1] == language:
                filePath = os.path.join(root,name)
                fileProcess(filePath, language, LM)
    # print(LM['uni'])

    #Save Model
    with open(fn_LM+'.pickle', 'wb') as handle:
        pickle.dump(LM, handle, protocol=pickle.HIGHEST_PROTOCOL)

    return LM


def fileProcess(filePath, language, LM):
    with open(filePath) as f:
        content = f.readlines()
    for sentence in content:
        sentence = preprocess(sentence, language)
        word_list = sentence.split()
        uniGramCount(word_list, LM)
        biGramCount(word_list, LM)
    # print(LM['uni'])
    # print(LM['bi'])


def uniGramCount(word_list, LM):
    for word in word_list:
        if word in LM['uni']:
            LM['uni'][word] += 1
        else:
            LM['uni'][word] = 1


def biGramCount(word_list, LM):
    sentence_len = len(word_list)
    for i in range(sentence_len - 1):
        if word_list[i] in LM['bi']: # check if the first word of bigram exist in the dic or not
            if word_list[i+1] in LM['bi'][word_list[i]]:
                LM['bi'][word_list[i]][word_list[i+1]] += 1
            else:
                LM['bi'][word_list[i]][word_list[i + 1]] = 1
        else:
            LM['bi'][word_list[i]] = {}
            LM['bi'][word_list[i]][word_list[i+1]] = 1

# if __name__ == "__main__":
#     LM = {}
#     LM['uni'] = {}
#     LM['bi'] = {}
#     data_dir = '/home/tianxiang/Desktop/CSC2511/A2_SMT/data/Hansard/Training/hansard.36.1.house.debates.001.e'
#     fileProcess(data_dir, 'e', LM)
#     print(LM['uni'])

    # data_dir = '/home/tianxiang/Desktop/CSC2511/A2_SMT/data/Hansard/Training'
    # lm_train(data_dir, 'e', './test')