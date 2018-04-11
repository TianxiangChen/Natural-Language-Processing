import re
import numpy as np
import os, fnmatch
from deepspeech.model import Model
import scipy.io.wavfile as wav
dataDir = '../data/'
# dataDir = '/u/cs401/A3/data/'

punc_set = set([',', '!', '"', '#', '$', '%', '&', '(', ')', '*', '+', '-', '.', '/', ':', ';', '<', '=', '>', '?', '@',
                '\\', '^', '_', '`', '{', '|', '}', '~', "'"])  # exclude [ and ] as required
pattern1 = re.compile("\[.*?\]")
pattern2 = re.compile("\<.*?\>")


def preProcess(strSent,deepspeech = False):
    '''
    pre-process the list of string, remove the punctuation and lowercase everything
    :param strSent: a string of sentence
    :return: a list of processed string
    '''
    # strSent = ' '.join([strList[0]] + strList[2:])
    if not deepspeech:
        strList = strSent.strip().lower().split()
        strSent = ' '.join(strList[2:])
    else:
        strSent = strSent.strip().lower()

    strSent = re.sub(pattern1, '', strSent)
    strSent = re.sub(pattern2, '', strSent)
    for punc in punc_set:
        strSent = strSent.replace(punc, '')

    return strSent.strip().split()


def Levenshtein(r, h):
    """
    Calculation of WER with Levenshtein distance.

    Works only for iterables up to 254 elements (uint8).
    O(nm) time ans space complexity.

    Parameters
    ----------
    r : list of strings
    h : list of strings

    Returns
    -------
    (WER, nS, nI, nD): (float, int, int, int) WER, number of substitutions, insertions, and deletions respectively

    """
    n = len(r)
    m = len(h)
    R = np.zeros((n + 1, m + 1))  # Matrix of distances
    B = np.zeros((n + 1, m + 1))  # Baktracking matrix, '1' for up, '2' for left, '3' for up-left

    R[:, 0] = np.arange(R.shape[0])
    R[0, :] = np.arange(R.shape[1])
    B[:, 0] = 1
    B[0, :] = 2
    B[0, 0] = 0

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            delVal = R[i - 1, j] + 1
            subVal = R[i - 1, j - 1] if r[i - 1] == h[j - 1] else R[i - 1, j - 1] + 1
            insVal = R[i, j - 1] + 1
            R[i, j] = min(delVal, subVal, insVal)
            if R[i, j] == delVal:
                B[i, j] = 1
            elif R[i, j] == insVal:
                B[i, j] = 2
            else:
                B[i, j] = 3

    accuracy = R[n, m] / n
    act_list = np.zeros((3))  # sub,ins,del
    i, j = n, m
    while i != 0 or j != 0:
        if B[i, j] == 3:  # up-left
            if R[i, j] == R[i - 1, j - 1] + 1:
                act_list[0] += 1
            i, j = i - 1, j - 1
        elif B[i, j] == 2:  # left
            j = j - 1
            act_list[1] += 1
        else:  # up
            i = i - 1
            act_list[2] += 1

    return accuracy, int(act_list[0]), int(act_list[1]), int(act_list[2])

if __name__ == "__main__":
    # print(Levenshtein("how to recognize speech".split(), "how to wreck a nice bench".split()))
    if os.path.isfile('bonus_deepspeech.txt'):
        os.remove('bonus_deepspeech.txt')

    pretrain_model = './speech/models/output_graph.pb'
    alphabet = './speech/models/alphabet.txt'
    ds = Model(pretrain_model, 26, 9, alphabet, 500)

    werGoogle = []
    werKaldi = []
    werDeep = []

    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print('Processing {}'.format(speaker))
            transcript = os.path.join(dataDir, speaker, 'transcripts.txt')
            transcriptG = os.path.join(dataDir, speaker, 'transcripts.Google.txt')
            transcriptK = os.path.join(dataDir, speaker, 'transcripts.Kaldi.txt')

            with open(transcript) as f:
                tContent = f.readlines()
            if not tContent:
                print('{} original transcript is empty. So skip it.'.format(speaker))
            else:
                gValid = False  # a flag to check if the transcipt has content
                kValid = False

                with open(transcriptG) as g:
                    gContent = g.readlines()
                if gContent:
                    gValid = True
                with open(transcriptK) as k:
                    kContent = k.readlines()
                if kContent:
                    kValid = True

                print('DeepSpeech Processing for {}'.format(speaker))
                files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*wav')
                files = sorted(files, key = lambda x: (x[1], x[0]))
                output_deepspeech = []
                numFiles = len(files)

                for i, file in enumerate(files):
                    print('DeepSpeech Processing for {}. {}/{}'.format(speaker, i, numFiles))
                    wav_file = os.path.join(dataDir, speaker, file)
                    print('File path: {}'.format(wav_file))
                    fs, audio = wav.read(wav_file)
                    recogData = ds.stt(audio, fs)
                    output_deepspeech.append(recogData)
                    print('DeepSpeech recognized content for  {}. {}/{}'.format(speaker, i, numFiles))
                    print(recogData)


                output = ''
                for i in range(len(tContent)):
                    args = Levenshtein(preProcess(tContent[i]), preProcess(output_deepspeech[i]))
                    output += '{:5} DeepS  {:2} {:.6f} S:{:3}, I:{:3}, D:{:3}\n'.format(speaker, i, args[0],
                            args[1], args[2], args[3])
                    werDeep.append(args[0])
                    if kValid:
                        args = Levenshtein(preProcess(tContent[i]), preProcess(kContent[i]))
                        output += '{:5} Kaldi  {:2} {:.6f} S:{:3}, I:{:3}, D:{:3}\n'.format(speaker, i, args[0],
                                args[1], args[2], args[3])
                        werKaldi.append(args[0])
                    if gValid:
                        args = Levenshtein(preProcess(tContent[i]), preProcess(gContent[i]))
                        output += '{:5} Google {:2} {:.6f} S:{:3}, I:{:3}, D:{:3}\n'.format(speaker, i, args[0],
                                args[1], args[2], args[3])
                        werGoogle.append(args[0])


                with open('bonus_deepspeech.txt', 'a') as f:
                    f.write(output)
    werG = np.array(werGoogle)
    werK = np.array(werKaldi)
    werD = np.array(werDeep)
    output = 'For Google: mean:{:.6f}, std:{:.6f}; For Kaldi: mean:{:.6f}, std:{:.6f}; For DeepS: mean:{:.6f}, std:{:.6f}\n'.format(
                    werG.mean(), werG.std(), werK.mean(), werK.std(), werD.mean(), werD.std())
    with open('bonus_deepspeech.txt', 'a') as f:
        f.write(output)