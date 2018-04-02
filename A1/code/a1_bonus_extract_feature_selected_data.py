import sys
import argparse
import os
import json

# My imports
import html
import re
import string
import spacy
import multiprocessing
import time
import numpy as np

indir = '/u/cs401/A1/data/'
feats_dir = '/u/cs401/A1/feats/'
wordlist_dir = '/u/cs401/Wordlists/'
# indir = '../data/'
# feats_dir = '../feats/'
# wordlist_dir = '../../Wordlists/'

nlp = spacy.load('en', disable=['parser', 'ner'])
alpha_set = set(string.ascii_lowercase + string.ascii_uppercase)
punctuation_set = set([',','!','"','#','$','%','&','(',')','*','+','-','.','/',':',';','<','=','>','?','@',
                    '[','\\',']','^','_','`','{','|','}','~'])

regex_punc = re.compile(r"['\,','\!','\"','\#','\$','\%','\&','\(','\)','\*','\+','\-','\.','\/','\:','\;','\<','\='," \
             r"'\>','\?','\@','\[','\\','\]','\^','\_','\`','\{','\|','\}','\~']+")
regex_http = re.compile(r"https?:\S+")
regex_www = re.compile(r"www\.\S+\.\S+")
Std_num = 999473181

def load_stopwords(wordlist_dir):
    stopwords_dir = wordlist_dir + 'StopWords'
    stopwords_set = set()
    with open(stopwords_dir) as f:
        for i in f:
            temp_list = i.split('\n')
            stopwords_set.add(temp_list[0])
    return stopwords_set
stopwords_set = load_stopwords(wordlist_dir)


def load_abbrv_set():
    '''
    This function loads abbrv. from provided files into a set.

    Parameter: None
    Return: abbrv_set: a set contains possible abbrv.s.
    '''
    abbrv_dir = []
    abbrv_dir.append(wordlist_dir + 'abbrev.english')
    abbrv_dir.append(wordlist_dir + 'pn_abbrev.english')
    abbrv_dir.append(wordlist_dir + 'pn_abbrev.english2')
    abbrv_set = set()

    for dir in abbrv_dir:
        with open(dir) as f:
            for i in f:
                temp_list = i.split('\n')
                abbrv_set.add(temp_list[0])
    return abbrv_set
abbrv_set = load_abbrv_set()# make it globally to save runtime


def punc_add_space(line):
    '''
    This function loads a line of words, and replace punctuations with whitespace

    Parameter: a line of words(string)
    Return: a line of words whose punctuations replaced by whitespace(string)
    '''
    # abbrv_set = load_abbrv_set()
    # punctuation_set = set(i for i in string.punctuation)
    # punctuation_set.remove('\'') # Remove the apostophes
    line = line.strip().split()
    newline = ''

    for word in line:
        if (not any(char in word for char in punctuation_set)) or (word in abbrv_set) :
            newline = newline + ' ' + word
        else:
            for abbrv in abbrv_set:
                word = word.replace(abbrv, ' ' + abbrv + ' ')

            # Split for multiple punctuation, also applies for single case
            relist = re.findall(regex_punc, word)
            if relist:
                for reg in relist:
                    word = word.replace(reg, ' ' + reg + ' ')
            newline = newline + ' ' + word
    newline = newline.strip()
    return newline


def preproc1( comment , steps=range(1,11)):
    ''' This function pre-processes a single comment

    Parameters:
        comment : string, the body of a comment
        steps   : list of ints, each entry in this list corresponds to a preprocessing step

    Returns:
        modComm : string, the modified comment
    '''

    modComm = ''
    modComm = comment
    if 1 in steps:
        # Remove all newline characters
        # I also remove all whitespace in the beginning and the end
        modComm = modComm.strip()
        modComm = modComm.replace('\n', ' ')
        modComm = " ".join(modComm.split())

    if 2 in steps:
        # Replace HTML charater codes (i.e. &..;) with their ASCII equivalent
        # https://docs.python.org/3/library/html.html
        modComm = html.unescape(modComm)
        modComm = " ".join(modComm.split())

    if 3 in steps:
        # Remove all URLs (i.e., tokens beginning with http or www)
        # https://docs.python.org/3/library/re.html
        modComm = re.sub(regex_http, " ", modComm) # Remove any string like 'http(s):xxx'
        modComm = re.sub(regex_www, " ", modComm) # Remove any string like 'www.XXX.XXX'
        modComm = " ".join(modComm.split())

    if 4 in steps:
        # Split each punctuation into its own token using whitespace except:
        #   Apostrophes
        #   Periods in abbreviation (e.g., e.g.) are not split from their tokens. E.g., e.g. stays e.g.
        #   Multiple punctuation (e.g., !?!, ...) are not split internally. E.g., Hi!!! becomes Hi !!!
        modComm = punc_add_space(modComm)
        modComm = " ".join(modComm.split())

    if 5 in steps:
        # Split clitics using whitespace
        # Clitics are contracted forms of words, such as n't, that are concatenated with the previous word
        # For all the cases where it is not "n't", basically we add a whitesapce before the "'"
        # Equivalently, we add whitespace for all "'", then remove for the "n't" case
        modComm = modComm.replace("'", " '")
        modComm = modComm.replace("n 't"," n't")
        modComm = " ".join(modComm.split())

    if 6 in steps:
        # Each token is tagge with its part-of-speech using spaCy
        # utt = nlp(u"{}".format(modComm))
        newComm = ''
        word_list = modComm.split()
        doc = spacy.tokens.Doc(nlp.vocab, words=word_list)
        doc = nlp.tagger(doc)
        for token in doc:
            newComm = newComm + ' ' + token.text + '/' + token.tag_
        modComm = " ".join(newComm.split())

    if 7 in steps:
        # Remove stopwords
        newComm = ''
        modComm_list = modComm.split()
        for word in modComm_list:
            word_list = word.rsplit('/', 1)
            if word_list[0] not in stopwords_set:
                newComm = newComm + ' ' + word
        modComm = " ".join(newComm.split())

    if 8 in steps:
        # Apply lemmatization using spaCy
        oldComm = ''
        modComm_list = modComm.split()
        for word in modComm_list:
            oldComm = oldComm + ' ' + word.rsplit('/',1)[0]

        word_list = oldComm.split()
        doc = spacy.tokens.Doc(nlp.vocab, words=word_list)
        doc = nlp.tagger(doc)
        modComm = ''

        for token in doc:
            if token.lemma_[0] == '-':
                modComm = modComm + ' ' + token.text + '/' + token.tag_
            else:
                modComm = modComm + ' ' + token.lemma_ + '/' + token.tag_
        modComm = " ".join(modComm.split())

    if 9 in steps:
        # Add a newline between each sentence
        newComm =''
        modComm_list = modComm.split()
        for word in modComm_list:
            if word != '':
                if word[-2:] == '/.':
                    newComm = newComm + ' ' + word + '\n '
                else:
                    newComm = newComm + ' ' + word
        modComm = newComm

    if 10 in steps:
        # Convert text to lowercase
        newComm = ''
        modComm_list = modComm.split(' ')
        for word in modComm_list:
            if word != '':
                word_list = word.rsplit('/', 1)
                newComm = newComm + ' ' + word_list[0].lower() + '/' + word_list[1]
        modComm = newComm

    return modComm


def process_single_file(file,subdir, queue):
    fullFile = os.path.join(subdir, file)
    print("Processing " + fullFile)

    data = json.load(open(fullFile))
    maxline = 10000
    data_proc = []
    start = Std_num % len(data)
    data_len = len(data)
    i = 0
    count = 0
    while count < maxline:
        j = json.loads(data[(start+i) % data_len])

        modComm = j['body'].strip()
        modComm = modComm.replace('\n', ' ')
        modComm = " ".join(modComm.split())

        if modComm != '' and modComm != '[removed]':
            post = {}
            post['body'] = preproc1(j['body'])
            post['id'] = j['id']
            post['cat'] = file
            count += 1
            print("Finish {}/{} -- {}".format(count, maxline, file))
            data_proc.append(post)
        i += 1
    queue.append(data_proc)
    return 0

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


def main( ):
    data = []
    print("Setp1 : Preprocess data")
    for subdir, dirs, files in os.walk(indir):
        result_list = multiprocessing.Manager().list()
        jobs = [multiprocessing.Process(target = process_single_file, args =(file, subdir, result_list )) for file in files]

        for job in jobs:
            job.start()

        for job in jobs:
            job.join()

        for result in result_list:
            data = data + result
        # print(result[0])

    print("Step2 : Feature Extraction")
    feats = np.zeros((len(data), 173 + 1))
    data_size = len(data)
    print("start loop")
    for i, reddit in enumerate(data):
        feats[i, 0:173] = extract1(reddit['body'])
        # print(reddit['body'])
        cat = reddit['cat']
        if cat == 'Alt':
            feats[i, 29:173] = alt_LIWC_features[alt_id_dic[reddit['id']]]
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
            print("for {}th datum from input file, the catagory {} is defined wrongly.".format(i, cat))
        print("{}/{} finish".format(i + 1, data_size))

    print("Done")
    np.savez_compressed('feats_selected.npz', feats)
    return 0

if __name__ == "__main__":
    main()
