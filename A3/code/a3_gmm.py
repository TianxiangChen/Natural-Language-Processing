from sklearn.model_selection import train_test_split
from scipy.misc import logsumexp
import numpy as np
import os, fnmatch
import random
import math

dataDir = '../data/'
# dataDir = '/u/cs401/A3/data/'

class theta:
    def __init__(self, name, M=8, d=13):
        self.name = name
        self.omega = np.zeros((M, 1))
        self.mu = np.zeros((M, d))
        self.Sigma = np.zeros((M, d))


# def log_b_m_x( m, x, myTheta, preComputedForM=[]):
#     ''' Returns the log probability of d-dimensional vector x using only component m of model myTheta
#         See equation 1 of the handout
#
#         As you'll see in tutorial, for efficiency, you can precompute something for 'm' that applies to all x outside of this function.
#         If you do this, you pass that precomputed component in preComputedForM
#
#     '''
#     sigma = (myTheta.Sigma)[m]
#     mu = (myTheta.mu)[m]
#     varInv = np.reciprocal(sigma)
#
#     term1 = np.multiply(np.square(x), varInv)
#     term2 = np.multiply(np.multiply(mu, x), varInv)
#     termDepend = np.sum(np.add (-1 * term1, term2), axis=0)
#     termIndepend = -1 * preComputedForM[m]
#
#     return termDepend + termIndepend

def log_b_m_X(m, X, myTheta):
    '''
    A modified version of the logb_m(x), vectorized computation for efficiency
    :param m: the index of GMM passed in for computation
    :param X: a numpy array of full input of one speaker(Txd)
    :param myTheta: a defined class for the GMM
    :return: (1xT) log(bm(xt)) for the specified m
    '''
    sigma = (myTheta.Sigma)[m]
    sigma_inv = np.reciprocal(sigma)
    mu = (myTheta.mu)[m]
    # pre-computed term for all elements
    term_fixed_1 = np.sum(np.multiply(np.square(mu), 0.5 * sigma_inv))
    term_fixed_2 = X.shape[1] / 2 * math.log(2 * math.pi)
    term_fixed_3 = 0.5 * np.log(np.prod(sigma))
    term_fixed = term_fixed_1 + term_fixed_2 + term_fixed_3

    logbm = np.sum(-0.5 * np.multiply(np.square(X), sigma_inv) + np.multiply(np.multiply(mu, X), sigma_inv), axis=1)

    return np.subtract(logbm, term_fixed)


# def log_p_m_x( m, x, myTheta, preComputedForM=[]):
#     ''' Returns the log probability of the m^{th} component given d-dimensional vector x, and model myTheta
#         See equation 2 of handout
#     '''
#     dimension = myTheta.omega.shape[0]
#     log_b = np.zeros((dimension,1))
#
#     for i in range(dimension):
#         log_b[i,0] = log_b_m_x( i, x, myTheta, preComputedForM)
#
#     numerator = myTheta.omega[m] * log_b[m]
#     denominator = np.sum(np.multiply(myTheta.omega, log_b))
#     return numerator / denominator

def log_p_m_X(myTheta, log_Bs):
    '''
    A modified version of the log(p(m|X)), vectorized computation for efficiency
    :param myTheta: a defined class for the GMM
    :param log_Bs: a precomputed (MxT) matrix, 'log_Bs', of log_b_m_x
    :return: vectorized log(p(m|X)), sized (MxT)
    '''
    term1 = np.add(log_Bs, np.log(myTheta.omega))
    term2 = logsumexp(log_Bs, b=myTheta.omega, axis=0)
    # weighed_sum = np.sum(weighed, axis=0)

    return np.subtract(term1, term2)


def logLik(log_Bs, myTheta):
    ''' Return the log likelihood of 'X' using model 'myTheta' and precomputed MxT matrix, 'log_Bs', of log_b_m_x

        X can be training data, when used in train( ... ), and
        X can be testing data, when used in test( ... ).

        We don't actually pass X directly to the function because we instead pass:

        log_Bs(m,t) is the log probability of vector x_t in component m, which is computed and stored outside of this function for efficiency.

        See equation 3 of the handout
    '''
    return np.sum(logsumexp(log_Bs, b=myTheta.omega, axis=0))


def train(speaker, X, M=8, epsilon=0.0, maxIter=20):
    ''' Train a model for the given speaker. Returns the theta (omega, mu, sigma)'''
    myTheta = theta(speaker, M, X.shape[1])
    print('Training the model for {}'.format(speaker))

    # initialize the model
    myTheta.omega.fill(1 / M)
    # randomly pick M vector from X for the intial mean
    randomArr = np.random.choice(X.shape[0], M, replace=False)
    for i in range(len(randomArr)):
        myTheta.mu[i] = X[randomArr[i]]
    myTheta.Sigma.fill(1)

    iter = 0
    prev_L = float('-inf')
    improvement = float('inf')
    T = X.shape[0]
    log_Bs = np.zeros((M, T))

    while iter <= maxIter and improvement >= epsilon:
        # ComputeIntermediateResults
        # print('iteration {}'.format(iter))
        for m in range(M):
            log_Bs[m, :] = log_b_m_X(m, X, myTheta)

        # calculate logLikelihood
        L = logLik(log_Bs, myTheta)

        # update parameters
        log_pmX_Temp = log_p_m_X(myTheta, log_Bs)
        prob_sum = np.exp(logsumexp(log_pmX_Temp, axis=1)).reshape(-1, 1)
        myTheta.omega = np.divide(prob_sum, T)
        myTheta.mu = np.divide(np.dot(np.exp(log_pmX_Temp), X), prob_sum)
        myTheta.Sigma = np.subtract(np.divide(np.dot(np.exp(log_pmX_Temp), np.square(X)), prob_sum),
                                    np.square(myTheta.mu))

        improvement = L - prev_L
        # print("L:{}, improvement:{}".format(L, improvement))
        prev_L = L
        iter += 1
    return myTheta


def test(mfcc, correctID, models, k=5):
    ''' Computes the likelihood of 'mfcc' in each model in 'models', where the correct model is 'correctID'
        If k>0, print to stdout the actual speaker and the k best likelihoods in this format:
               [ACTUAL_ID]
               [SNAME1] [LOGLIK1]
               [SNAME2] [LOGLIK2]
               ...
               [SNAMEK] [LOGLIKK]

        e.g.,
               S-5A -9.21034037197
        the format of the log likelihood (number of decimal places, or exponent) does not matter
    '''
    bestModel = -1
    numSpeaker = len(models)
    logLik = np.zeros((numSpeaker))
    T = mfcc.shape[0]
    M = models[0].omega.shape[0]
    logBs = np.zeros((numSpeaker, M, T))
    for i in range(numSpeaker):
        for m in range(M):
            logBs[i, m, :] = log_b_m_X(m, mfcc, models[i])

    for i in range(numSpeaker):
        logLik[i] = np.sum(logsumexp(logBs[i], b=models[i].omega, axis=0))
    bestModel = np.argmax(logLik)
    if k > 0:
        topK = logLik.argsort()[-k:][::-1]
        output = '{}\n'.format(models[correctID].name)
        for i in range(k):
            output += '{:5} {}\n'.format(models[int(topK[i])].name, logLik[int(topK[i])])
        print(output)
        with open('gmmLiks.txt', 'a') as f:
            f.write(output)
    return 1 if (bestModel == correctID) else 0


if __name__ == "__main__":

    trainThetas = []
    testMFCCs = []
    # print('TODO: you will need to modify this main block for Sec 2.3')
    d = 13
    k = 5  # number of top speakers to display, <= 0 if none
    M = 8
    epsilon = 0.0
    maxIter = 20

    # train a model for each speaker, and reserve data for testing
    for subdir, dirs, files in os.walk(dataDir):
        for speaker in dirs:
            print(speaker)

            files = fnmatch.filter(os.listdir(os.path.join(dataDir, speaker)), '*npy')
            random.shuffle(files)

            testMFCC = np.load(os.path.join(dataDir, speaker, files.pop()))
            testMFCCs.append(testMFCC)

            X = np.empty((0, d))
            for file in files:
                myMFCC = np.load(os.path.join(dataDir, speaker, file))
                X = np.append(X, myMFCC, axis=0)

            trainThetas.append(train(speaker, X, M, epsilon, maxIter))

    # evaluate
    numCorrect = 0;
    if os.path.isfile('gmmLiks.txt'):
        os.remove('gmmLiks.txt')
    for i in range(0, len(testMFCCs)):
        numCorrect += test(testMFCCs[i], i, trainThetas, k)
    accuracy = 1.0 * numCorrect / len(testMFCCs)
    print("Accurancy is {}".format(accuracy))