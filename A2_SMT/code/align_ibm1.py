from lm_train import *
from log_prob import *
from preprocess import *
import pickle
from math import log
import os

def align_ibm1(train_dir, num_sentences, max_iter, fn_AM):
    """
	Implements the training of IBM-1 word alignment algoirthm. 
	We assume that we are implemented P(foreign|english)
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	max_iter : 		(int) the maximum number of iterations of the EM algorithm
	fn_AM : 		(string) the location to save the alignment model
	
	OUTPUT:
	AM :			(dictionary) alignment model structure
	
	The dictionary AM is a dictionary of dictionaries where AM['english_word']['foreign_word'] 
	is the computed expectation that the foreign_word is produced by english_word.
	
			LM['house']['maison'] = 0.5
	"""
    AM = {}
    
    # Read training data
    data = read_hansard(train_dir, num_sentences)
    # print(data['e'])
    # print(data['f'])

    # Initialize AM uniformly
    AM = initialize(data['e'], data['f'], AM)

    # Iterate between E and M steps
    AM = em_step(max_iter, data['e'], data['f'], AM)

    # Save Model
    with open(fn_AM + '.pickle', 'wb') as handle:
        pickle.dump(AM, handle, protocol=pickle.HIGHEST_PROTOCOL)
    return AM
    
# ------------ Support functions --------------
def readLine(eContent, fContent, numLineRead, data):
    count = 0
    while count < numLineRead:
        data['e'].append(preprocess(eContent[count], 'e').split())
        data['f'].append(preprocess(fContent[count], 'f').split())
        count += 1
    return data


def read_hansard(train_dir, num_sentences):
    """
	Read up to num_sentences from train_dir.
	
	INPUTS:
	train_dir : 	(string) The top-level directory name containing data
					e.g., '/u/cs401/A2_SMT/data/Hansard/Testing/'
	num_sentences : (int) the maximum number of training sentences to consider
	
	
	Make sure to preprocess!
	Remember that the i^th line in fubar.e corresponds to the i^th line in fubar.f.
	
	Make sure to read the files in an aligned manner.
	"""
    data = {'e':[], 'f':[]}
    lineCount = 0

    for root, dirs, files in os.walk(train_dir):
        filesFiltered = [file for file in files if file[-2:] == '.e' or file[-2:] == '.f']
        fileNameDic = {}
        fileNameList = []

        for file in filesFiltered:
            if file[:-2] in fileNameDic:
                fileNameList.append(file[:-2])
            else:
                fileNameDic[file[:-2]] = 1
        fileNameList = sorted(fileNameList, key = lambda s: s.lower())

        for name in fileNameList:
            # print(name)
            eName = name + '.e'
            fName = name + '.f'
            eFilePath = os.path.join(root,eName)
            fFilePath = os.path.join(root,fName)

            with open(eFilePath) as f:
                eContent = f.readlines()
            with open(fFilePath) as f:
                fContent = f.readlines()

            if(len(eContent) + lineCount > num_sentences):
                numLineRead = num_sentences - lineCount
                data = readLine(eContent, fContent, numLineRead, data)
                print('Finish reading data')
                return data
            data = readLine(eContent, fContent, len(eContent), data)
            lineCount += len(eContent)

    print('Finish reading data')
    return data


def initialize(eng, fre, AM):
    """
	Initialize alignment model uniformly.
	Only set non-zero probabilities where word pairs appear in corresponding sentences.
	"""
    wordPosDic = {} # indicate the position(line no.) where the en word is in the french sentence list
    for i,line in enumerate(eng):
        for word in line:
            if word not in wordPosDic:
                wordPosDic[word] = []
            wordPosDic[word].append(i)

    for line in eng:
        for word in line:
            if word not in AM and word != 'SENTSTART' and word != 'SENTEND':
                AM[word] = {}
                frenchWordSet = set([])
                for existLine in wordPosDic[word]:
                    for fWord in fre[existLine]:
                        if fWord != 'SENTSTART' and fWord != 'SENTEND':
                            frenchWordSet.add(fWord)

                uniProb = 1 / len(frenchWordSet)
                for uniqueFrenchWord in frenchWordSet:
                    AM[word][uniqueFrenchWord] = uniProb

    # Reset SENTSTART and SENTEND prob to 1, avoid any alignment between start/end sign and actual word
    AM['SENTSTART'] = {}
    AM['SENTEND'] = {}
    AM['SENTSTART']['SENTSTART'] = 1
    AM['SENTEND']['SENTEND'] = 1
    print('Finish initializing the AM')
    return AM

    
def em_step(t, eng, fre, AM):
    """
	One step in the EM algorithm.
	Follows the pseudo-code given in the tutorial slides.
	"""
    for counter in range(t):
        print('EM round {} / round {}'.format(counter, t))
        tCount = {}
        total = {}

        for firstEle in AM:
            if firstEle != 'SENTSTART' and firstEle != 'SENTEND':
                tCount[firstEle] = {}
                total[firstEle] = 0
                for secondEle in AM[firstEle]:
                    if secondEle != 'SENTSTART' and firstEle != 'SENTEND':
                        tCount[firstEle][secondEle] = 0
            # print(tCount)
            # print(total)
        for i, fSent in enumerate(fre):
            fSentence = fSent[:]
            fSentence.remove('SENTSTART')
            fSentence.remove('SENTEND')
            fSentSet = set(fSentence)
            for uniWordf in fSentSet:
                denom_c = 0
                eSentence = eng[i][:]
                eSentence.remove('SENTSTART')
                eSentence.remove('SENTEND')
                eSentSet = set(eSentence)
                for uniWorde in eSentSet:
                    # print(uniWorde in AM)
                    # print(uniWordf in AM[uniWorde])
                    # print(AM[uniWorde][uniWordf])
                    # print(fSentence.count(uniWordf))
                    denom_c += AM[uniWorde][uniWordf] * fSentence.count(uniWordf)
                for uniWorde in eSentSet:
                    tCount[uniWorde][uniWordf] += AM[uniWorde][uniWordf] * fSentence.count(uniWordf) * eSentence.count(uniWorde) / denom_c
                    total[uniWorde] += AM[uniWorde][uniWordf] * fSentence.count(uniWordf) * eSentence.count(uniWorde) / denom_c

        for e in total:
            for f in tCount[e]:
                AM[e][f] = tCount[e][f] / total[e]

    return AM


# if __name__ == "__main__":
#     # train_dir = '/home/tianxiang/Desktop/CSC2511/A2_SMT/data/Hansard/Testing'
#     train_dir = '/home/tianxiang/Desktop/CSC2511/A2_SMT/data/Toy/'
#     num_sentences = 10
#     max_iter = 10000
#     fn_AM = 'aa'
#     res = align_ibm1(train_dir, num_sentences, max_iter, fn_AM)
#     print(res)