import numpy as np
import sys
import argparse
import os

# My imports
import json
import csv

# indir = '/u/cs401/A1/data/'
# feats_dir = '/u/cs401/A1/feats/'
# wordlist_dir = '/u/cs401/Wordlists/'
indir = '../data/'
feats_dir = '../feats/'
wordlist_dir = '../../Wordlists/'

first_person_set = set(['I', 'me', 'my', 'mine', 'we', 'us', 'our', 'ours'])
second_persion_set = set(['you', 'your', 'yours', 'u', 'ur', 'urs'])
third_person_set = set(['he', 'him', 'his', 'she', 'her', 'hers', 'it', 'its', 'they', 'them', 'their', 'theirs'])
future_tense_set = set(["’ll", 'will', 'gonna']) # also going+to+VB
common_nouns_set = set(['NN', 'NNS'])
proper_nouns_set = set(['NNP', 'NNPS'])
adverbs_set = set(['RB', 'RBR', 'RBS'])
wh_words_set = set(['WDT', 'WP', 'WP$', 'WRB'])
modern_slang_acronyms_set = set(['smh', 'fwb', 'lmfao', 'lmao', 'lms', 'tbh', 'rofl', 'wtf', 'bff', 'wyd',
'lylc', 'brb', 'atm', 'imao', 'sml', 'btw','bw', 'imho', 'fyi', 'ppl', 'sob', 'ttyl', 'imo',
'ltr', 'thx', 'kk', 'omg', 'omfg', 'ttys', 'afn', 'bbs', 'cya', 'ez', 'f2f', 'gtr', 'ic',
'jk', 'k', 'ly', 'ya', 'nm', 'np', 'plz', 'ru', 'so', 'tc', 'tmi', 'ym', 'ur', 'u', 'sol', 'fml'])
punctuation_tag_set = set(['#', '$', '.', ',', ':', '(', ')', '"', '‘', '“', '’', '”'])
punctuation_set = set([',','!','"','#','$','%','&','(',')','*','+','-','.','/',':',';','<','=','>','?','@',
                    '[','\\',']','^','_','`','{','|','}','~'])


def sentence_split(comment):
    origin_sentence = []
    tags = []
    word_list = comment.split()
    for word in word_list:
        temp_list = word.rsplit('/', 1)
        origin_sentence.append(temp_list[0])
        tags.append(temp_list[1])
    return origin_sentence, tags

def read_BristolGilhoolyLogie(wordlist_dir):
    BristolGilhoolyLogie_dic = {}
    filename = wordlist_dir + 'BristolNorms+GilhoolyLogie.csv'

    with open(filename, 'r') as csvfile:
        header_line = next(csvfile)
        for line in csvfile.readlines():
            col = line.split(',')
            info = []
            info.extend((col[3], col[4], col[5]))
            BristolGilhoolyLogie_dic[col[1]] = info
    return BristolGilhoolyLogie_dic
BristolGilhoolyLogie_dic = read_BristolGilhoolyLogie(wordlist_dir)


def read_Warriner(wordlist_dir):
    Warriner_dic = {}
    filename = wordlist_dir + 'Ratings_Warriner_et_al.csv'

    with open(filename, 'r') as csvfile:
        header_line = next(csvfile)
        for line in csvfile.readlines():
            col = line.split(',')
            info = []
            info.extend((col[2], col[5], col[8]))
            BristolGilhoolyLogie_dic[col[1]] = info
    return BristolGilhoolyLogie_dic
Warriner_dic = read_Warriner(wordlist_dir)


def create_dic(filename):
    dic = {}
    with open(filename) as f:
        line = 1
        for i in f:
            temp_list = i.split('\n')
            dic[temp_list[0]] = line
            line += 1
    return dic


def load_LIWC_ids(feats_dir):
    alt_filename = feats_dir + 'Alt_IDs.txt'
    left_filename = feats_dir + 'Left_IDs.txt'
    right_filename = feats_dir + 'Right_IDs.txt'
    center_filename = feats_dir + 'Center_IDs.txt'

    alt_id_dic = create_dic(alt_filename)
    left_id_dic = create_dic(left_filename)
    right_id_dic = create_dic(right_filename)
    center_id_dic = create_dic(center_filename)

    return alt_id_dic, left_id_dic, right_id_dic, center_id_dic
alt_id_dic, left_id_dic, right_id_dic, center_id_dic = load_LIWC_ids(feats_dir)


def load_LIWC_feats(feats_dir):
    alt_filename = feats_dir + 'Alt_feats.dat.npy'
    left_filename = feats_dir + 'Left_feats.dat.npy'
    right_filename = feats_dir + 'Right_feats.dat.npy'
    center_filename = feats_dir + 'Center_feats.dat.npy'

    alt_LIWC_features = np.load(alt_filename)
    left_LIWC_features = np.load(left_filename)
    right_LIWC_features = np.load(right_filename)
    center_LIWC_features = np.load(center_filename)

    return alt_LIWC_features, left_LIWC_features, right_LIWC_features, center_LIWC_features
alt_LIWC_features, left_LIWC_features, right_LIWC_features, center_LIWC_features = load_LIWC_feats(feats_dir)


def count_future_tense(origin_sentence, tags):
    counter = 0
    length = len(origin_sentence)
    for i, word in enumerate(origin_sentence):
        if word == "'ll" or word == 'will' or word == 'gonna':
            if i < length - 1: # not srue how python implment multi-logical expression, for safety writing nested to
                # avoid out of range
                if tags[i+1] == 'VB':
                    counter += 1
        elif word == 'going':
            if i < length - 2:
                if origin_sentence[i+1] == 'to' and tags[i+2] == 'VB':
                    counter += 1
    return counter

def multi_char_punc_count(origin_sentence):
    counter = 0
    for word in origin_sentence:
        if set(word) <= punctuation_set and len(word) > 1:
            counter += 1
    return counter


def uppercase_word_count(origin_sentence):
    counter = 0
    for word in origin_sentence:
        if word.isupper() and len(word) >= 3:
            counter += 1
    return counter

def average_sentence_length(comment):
    wordlist = comment.split(' ')
    sentence = []
    counter = 0
    for word in wordlist:
        counter += 1
        if word[-1:] == '\n':
            if counter != 1:
                sentence.append(counter)
                counter = 0
    return np.mean(sentence) if len(sentence) > 0 else 0


def average_token_length(origin_sentence,tags):
    token_len = []

    for i, word in enumerate(origin_sentence):
        if tags[i] not in punctuation_tag_set:
            token_len.append(len(word))
    return np.mean(token_len) if len(token_len) > 0 else 0


def BristolGilhoolyLogie_avg_std(origin_sentence,item):
    data = []
    for word in origin_sentence:
        if word in BristolGilhoolyLogie_dic:
            data.append(float(BristolGilhoolyLogie_dic[word][item]))
    if len(data) > 0:
        return np.mean(data), np.std(data)
    else:
        return 0, 0


def Warriner_avg_std(origin_sentence, item):
    data = []
    for word in origin_sentence:
        if word in Warriner_dic:
            data.append(float(Warriner_dic[word][item]))
    if len(data) > 0:
        return np.mean(data), np.std(data)
    else:
        return 0, 0


def extract1( comment ):
    ''' This function extracts features from a single comment

    Parameters:
        comment : string, the body of a comment (after preprocessing)

    Returns:
        feats : numpy Array, a 173-length vector of floating point features (only the first 29 are expected to be filled, here)
    '''

    # TODO: your code here
    feat = np.zeros((173))
    comment_new = ' '.join(comment.split())
    if comment_new == '': # should i handle [removed] here?
        print("empty_comment")
        return feat

    origin_sentence, tags = sentence_split(comment)

    feat[0] = sum(1 for s in origin_sentence if s in first_person_set)
    feat[1] = sum(1 for s in origin_sentence if s in second_persion_set)
    feat[2] = sum(1 for s in origin_sentence if s in third_person_set)
    feat[3] = sum(1 for s in tags if s == 'CC')
    feat[4] = sum(1 for s in tags if s == 'VBD')
    feat[5] = count_future_tense(origin_sentence, tags)
    feat[6] = sum(1 for s in tags if s == ',')
    feat[7] = multi_char_punc_count(origin_sentence)
    feat[8] = sum(1 for s in tags if s in common_nouns_set)
    feat[9] = sum(1 for s in tags if s in proper_nouns_set)
    feat[10] = sum(1 for s in tags if s in adverbs_set)
    feat[11] = sum(1 for s in tags if s in wh_words_set)
    feat[12] = sum(1 for s in origin_sentence if s in modern_slang_acronyms_set)
    feat[13] = uppercase_word_count(origin_sentence)
    feat[14] = average_sentence_length(comment)
    feat[15] = average_token_length(origin_sentence,tags)
    feat[16] = comment.count('\n')
    feat[17],feat[20] = BristolGilhoolyLogie_avg_std(origin_sentence,0)
    feat[18],feat[21] = BristolGilhoolyLogie_avg_std(origin_sentence, 1)
    feat[19],feat[22] = BristolGilhoolyLogie_avg_std(origin_sentence, 2)
    feat[23],feat[26] = Warriner_avg_std(origin_sentence, 0)
    feat[24],feat[27] = Warriner_avg_std(origin_sentence, 1)
    feat[25],feat[28] = Warriner_avg_std(origin_sentence, 2)

    return feat

def main( args ):

    data = json.load(open(args.input))
    feats = np.zeros( (len(data), 173+1) )
    data_size = len(data)
    print("start loop")
    # TODO: your code here
    for i, reddit in enumerate(data):
        feats[i,0:173] = extract1(reddit['body'])
        # print(reddit['body'])
        cat = reddit['cat']
        if cat == 'Alt':
            feats[i,29:173] = alt_LIWC_features[alt_id_dic[reddit['id']]]
            feats[i, -1] = 3
        elif cat == 'Left':
            feats[i, 29:173] = left_LIWC_features[left_id_dic[reddit['id']]]
            feats[i, -1] = 0
        elif cat == 'Right':
            feats[i, 29:173] = right_LIWC_features[right_id_dic[reddit['id']]]
            feats[i, -1] = 2
        elif cat == 'Center':
            feats[i, 29:173] = center_LIWC_features[center_id_dic[reddit['id']]]
            feats[i, -1] = 1
        else:
            print("for {}th datum from input file, the catagory {} is defined wrongly.".format(i,cat))

        print("{}/{} finish".format(i+1, data_size))

    print("Done")
    np.savez_compressed( args.output, feats)

    
if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("-i", "--input", help="The input JSON file, preprocessed as in Task 1", required=True)
    args = parser.parse_args()
                 

    main(args)

