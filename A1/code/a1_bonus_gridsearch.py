from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
import numpy as np
import argparse
import sys
import os
import csv
from datetime import datetime




def classify_func(X_train, X_test, y_train, y_test, i):
    if i == 1:
        parameters = {'alpha': [0.025, 1]}
        MLP = MLPClassifier()
        print("finding..")
        clf = GridSearchCV(MLP, parameters)
        clf.fit(X_train, y_train)
        alpha_opt = clf.best_params_['alpha']
        print("Best alpha: {}".format(alpha_opt))
        accuracy_value = clf.score(X_test, y_test)
        print(accuracy_value)

    elif i == 2:
        parameters = {'n_estimators': [50, 100]}
        AB_clf = AdaBoostClassifier()
        print("finding..")
        clf = GridSearchCV(AB_clf, parameters)
        clf.fit(X_train, y_train)
        alpha_opt = clf.best_params_['n_estimators']
        print("Best n_estimators: {}".format(alpha_opt))
        accuracy_value = clf.score(X_test, y_test)
        print(accuracy_value)

    return 0


def classify_gridsearch(filename):
    print('Processing')
    features = np.load(filename)['arr_0']
    X_train, X_test, y_train, y_test = train_test_split(features[:, :173], features[:, 173], test_size=0.2)

    for i in range(1, 3):
        classify_func(X_train, X_test, y_train, y_test, i)
    return 0




if __name__ == "__main__":
    startTime = datetime.now()
    print("Using posts including empty and removed")
    classify_gridsearch('feats.npz')
    print("Using posts without empty and removed")
    classify_gridsearch('feats_selected.npz')
    print("Total runtime: {}".format(datetime.now() - startTime))
