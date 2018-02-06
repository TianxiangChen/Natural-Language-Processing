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

# indir = '/u/cs401/A1/data/';
# wordlist_dir = '/u/cs401/Wordlists/'
indir = '../data/';
wordlist_dir = '../../Wordlists/'
nlp = spacy.load('en', disable=['parser', 'ner'])
alpha_list = list(string.ascii_lowercase) + list(string.ascii_uppercase)
punctuation_list = [',','!','"','#','$','%','&','(',')','*','+','-','.','/',':',';','<','=','>','?','@',
                    '[','\\',']','^','_','`','{','|','}','~']

regex_punc = re.compile(r"['\,','\!','\"','\#','\$','\%','\&','\(','\)','\*','\+','\-','\.','\/','\:','\;','\<','\='," \
             r"'\>','\?','\@','\[','\\','\]','\^','\_','\`','\{','\|','\}','\~']+")
regex_http = re.compile(r"https?:\S+")
regex_www = re.compile(r"www\.\S+\.\S+")

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
        if (not any(char in word for char in punctuation_list)) or (word in abbrv_set) :
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
        utt = nlp(u"{}".format(modComm))
        newComm = ''

        for token in utt:
            newComm = newComm + ' ' + token.text + '/' + token.tag_
        modComm = newComm.strip()
        modComm = " ".join(modComm.split())

    if 7 in steps:
        # Remove stopwords
        stopwords_dir = wordlist_dir + 'StopWords'
        stopwords_set = set()
        with open(stopwords_dir) as f:
            for i in f:
                temp_list = i.split('\n')
                stopwords_set.add(temp_list[0])

        newComm = ''
        modComm_list = modComm.split(" ")
        for word in modComm_list:
            word_list = word.rsplit('/', 1)
            if word_list[0] not in stopwords_set:
                newComm = newComm + ' ' + word

        modComm = newComm.strip()
        modComm = " ".join(modComm.split())

    if 8 in steps:
        # Apply lemmatization using spaCy
        oldComm = ''
        modComm_list = modComm.split(" ")
        for word in modComm_list:
            oldComm = oldComm + ' ' + word.rsplit('/',1)[0]

        oldComm = oldComm.strip()
        utt = nlp(u"{}".format(oldComm))
        modComm = ''

        for token in utt:
            if token.lemma_[0] == '-':
                modComm = modComm + ' ' + token.text + '/' + token.tag_
            else:
                modComm = modComm + ' ' + token.lemma_ + '/' + token.tag_
        modComm = modComm.strip()
        modComm = " ".join(modComm.split())

    if 9 in steps:
        # Add a newline between each sentence
        newComm =''
        modComm_list = modComm.split(" ")
        for word in modComm_list:
            if word[-2:] == '/.':
                newComm = newComm + ' ' + word + '\n '
            else:
                newComm = newComm + ' ' + word
        modComm = newComm
    if 10 in steps:
        # Convert text to lowercase
        newComm = ''
        modComm_list = modComm.split(" ")
        for word in modComm_list:
            if any(ch in word for ch in alpha_list):
                word_list = word.rsplit('/', 1)
                newComm = newComm + ' ' + word_list[0].lower() + '/' + word_list[1]
            else:
                newComm = newComm + ' ' + word


    return modComm

def process_single_file(file,subdir, args, queue):
    fullFile = os.path.join(subdir, file)
    print("Processing " + fullFile)

    data = json.load(open(fullFile))
    maxline = args.max if len(data) >= args.max else len(data)
    data_proc = []
    start = args.ID[0] % len(data)
    data_len = len(data)
    for i in range(maxline):
        j = json.loads(data[(start+i) % data_len])
        post = {}
        post['body'] = preproc1(j['body'])
        post['cat'] = file
        print("Finish {}/{} -- {}".format(i + 1, maxline, file))
        data_proc.append(post)
        # time.sleep(3)
    queue.append(data_proc)
    return 0


def main( args ):

    allOutput = []
    feature_list = ['ups', 'downs', 'score', 'controversiality', 'subreddit', 'author', 'body','id']

    for subdir, dirs, files in os.walk(indir):
        result_list = multiprocessing.Manager().list()
        jobs = [multiprocessing.Process(target = process_single_file, args =(file, subdir, args, result_list )) for file in files]

        for job in jobs:
            job.start()

        for job in jobs:
            job.join()

        for result in result_list:
            allOutput = allOutput + result
        # print(result[0])

    # print(len(allOutput))
    fout = open(args.output, 'w')
    fout.write(json.dumps(allOutput))
    fout.close()
    return 0

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument('ID', metavar='N', type=int, nargs=1,
                        help='your student ID')
    parser.add_argument("-o", "--output", help="Directs the output to a filename of your choice", required=True)
    parser.add_argument("--max", help="The maximum number of comments to read from each file", default=10000)
    args = parser.parse_args()

    if (args.max > 200272):
        print ("Error: If you want to read more than 200,272 comments per file, you have to read them all.")
        sys.exit(1)

    main(args)
