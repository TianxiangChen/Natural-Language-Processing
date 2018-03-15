from preprocess import *
from lm_train import *
from log_prob import *
from align_ibm1 import *
from decode import *
from BLEU_score import *
import os
import pickle

def processFile(file_path, language):
    with open(file_path) as f:
        content = f.readlines()
    processed_list = []
    for line in content:
        processed_list.append(preprocess(line, language))
    return processed_list


def resEvalTranslate(transSentList, eng_ans_dir, eng_ans_google_dir):
    ansE = processFile(eng_ans_dir, 'e')
    ansGoogle = processFile(eng_ans_google_dir, 'e')

    evalRes = []
    for i, line in enumerate(transSentList):
        ref = [ansE[i], ansGoogle[i]]
        score = []
        for n in range(1,4):
            score.append(BLEU_score(line, ref, n))
        evalRes.append(score)
    return evalRes


def generateLM(filename, language, train_dir):
    fullname = filename + '.pickle'
    if os.path.isfile(fullname):
        with (open(fullname, "rb")) as openfile:
            LM = pickle.load(openfile)
    else:
        print(train_dir)
        LM = lm_train(train_dir, language, filename)
    return LM


def generateAM(filename, train_dir, num_sent, iter):
    fullname = filename + '.pickle'
    if os.path.isfile(fullname):
        with (open(fullname, "rb")) as openfile:
            AM = pickle.load(openfile)
    else:
        AM = align_ibm1(train_dir, num_sent, iter, filename)
    return AM



if __name__ == "__main__":
    max_iter = 5
    train_dir = '../data/Hansard/Training/'
    test_dir = '../data/Hansard/Testing/'
    fre_test_dir = '../data/Hansard/Testing/Task5.f'
    eng_ans_dir = '../data/Hansard/Testing/Task5.e'
    eng_ans_google_dir = '../data/Hansard/Testing/Task5.google.e'

    LM = generateLM('LM_e','e', train_dir)

    AM_1k = generateAM('AM_1k', train_dir, 1000, max_iter)
    AM_10k = generateAM('AM_10k', train_dir, 10000, max_iter)
    AM_15k = generateAM('AM_15k', train_dir, 15000, max_iter)
    AM_30k = generateAM('AM_30k', train_dir, 30000, max_iter)
    AM_model = [AM_1k, AM_10k, AM_15k, AM_30k]

    fre_proc_list = processFile(fre_test_dir, 'f')

    decoded_sent = []
    for model in AM_model:
        decoded_sent_by_model = []
        for fre_sent in fre_proc_list:
            decoded_sent_by_model.append(decode(fre_sent, LM, model))
        decoded_sent.append(decoded_sent_by_model)

    # print(len(decoded_sent[0]))
    # print(decoded_sent[0][0])

    results = []
    for i in range(len(AM_model)):
        # print(sent_list)
        results.append(resEvalTranslate(decoded_sent[i], eng_ans_dir, eng_ans_google_dir))

    if os.path.isfile('Task5.txt'):
        os.remove('Task5.txt')
    with open('Task5.txt', 'a') as f:
        for i in range(len(results[0])):
            f.write('Sentence {:02} : '.format(i+1))
            for j in range(len(AM_model)):
                f.write('{}'.format(results[j][i]))
            f.write('\n')
        f.write('\nThe BLEU score for each test sentence is reported in the order of with 1K, 10K, 15K and 30K data trained alignment models.\n')
        f.write('Inside each bracket, the three BLEU scores are for n = 1,2,3.\n')
        f.write('\nFrom the result above, we can find that, generally, as the larger size of the data for the alignment model, the better result (higher BLEU) it is.\n')
        f.write('This is common for most training problems since the larger size of the data helps to characterize the model better.\n')
        f.write('Also, for the same alignment model, the BLEU score for n = 1 is higher than n = 2, and later is higher than n = 3.\n')
        f.write('This is also easy for understanding since it is easier for matching the unigram (n = 1) than the bigram and trigram (n = 2,3)')