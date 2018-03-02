from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC, SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
from sklearn.metrics import confusion_matrix
from scipy import stats
import numpy as np
import argparse
import sys
import os
import csv
from datetime import datetime


def accuracy( C ):
    ''' Compute accuracy given Numpy array confusion matrix C. Returns a floating point value '''
    return np.sum(np.diagonal(C)) / np.sum(C)

def recall( C ):
    ''' Compute recall given Numpy array confusion matrix C. Returns a list of floating point values '''
    dimension = len(C)
    recall = []
    for i in range(dimension):
        recall.append(C[i, i] / np.sum(C[:, i]))
    return recall


def precision( C ):
    ''' Compute precision given Numpy array confusion matrix C. Returns a list of floating point values '''
    dimension = len(C)
    precision = []
    for i in range(dimension):
        precision.append(C[i, i] / np.sum(C[i, :]))
    return precision
    
def generatre_result_for_clssifier(classid, c_matrix, class_name):
    accuracy_value = accuracy(c_matrix)
    recall_value = recall(c_matrix)
    precision_value = precision(c_matrix)
    print("Method {}: {}\nAccuracy: {}\nRecall: {}\nPrecision: {}\n".format(classid,class_name,accuracy_value,recall_value,precision_value))
    result = [classid, accuracy_value] + recall_value + precision_value + c_matrix.flatten().tolist()
    return accuracy_value, result

def classify_func(X_train, X_test, y_train, y_test, i):
    if i == 1:
        LinearSVC_clf = LinearSVC()
        LinearSVC_clf.fit(X_train, y_train)
        c_matrix = confusion_matrix(y_test, LinearSVC_clf.predict(X_test))
        accuracy_value, result = generatre_result_for_clssifier(i, c_matrix, 'LinearSVC')
    elif i == 2:
        SVC_clf = SVC(gamma=2)
        SVC_clf.fit(X_train, y_train)
        c_matrix = confusion_matrix(y_test, SVC_clf.predict(X_test))
        accuracy_value, result = generatre_result_for_clssifier(i, c_matrix, 'SVC with Gamma = 2')
    elif i == 3:
        RF_clf = RandomForestClassifier(n_estimators=10, max_depth=5)
        RF_clf.fit(X_train, y_train)
        c_matrix = confusion_matrix(y_test, RF_clf.predict(X_test))
        accuracy_value, result = generatre_result_for_clssifier(i, c_matrix, 'RandomForestClassifier')
    elif i == 4:
        MLP_clf = MLPClassifier(alpha=0.05)
        MLP_clf.fit(X_train, y_train)
        c_matrix = confusion_matrix(y_test, MLP_clf.predict(X_test))
        accuracy_value, result = generatre_result_for_clssifier(i, c_matrix, 'MLPClassifier')
    elif i == 5:
        AB_clf = AdaBoostClassifier()
        AB_clf.fit(X_train, y_train)
        c_matrix = confusion_matrix(y_test, AB_clf.predict(X_test))
        accuracy_value, result = generatre_result_for_clssifier(i, c_matrix, 'AdaBoostClassifier')
    return accuracy_value, result


def class31(filename):
    ''' This function performs experiment 3.1
    
    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:      
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier
    '''
    print('Processing Section 3.1')
    if os.path.isfile('a1_3.1.csv'):
        os.remove('a1_3.1.csv')

    features = np.load(filename)['arr_0']
    accuracy_list = []
    results = []
    X_train, X_test, y_train, y_test = train_test_split(features[:, :173], features[:, 173], test_size=0.2)

    for i in range(1,6):
        accuracy_value, result = classify_func(X_train, X_test, y_train, y_test, i)
        accuracy_list.append(accuracy_value)
        results.append(result)


    with open('a1_3.1.csv', 'w',newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerows(results)
    iBest = accuracy_list.index(max(accuracy_list)) + 1
    return (X_train, X_test, y_train, y_test,iBest)


def class32(X_train, X_test, y_train, y_test,iBest):
    ''' This function performs experiment 3.2
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  

    Returns:
       X_1k: numPy array, just 1K rows of X_train
       y_1k: numPy array, just 1K rows of y_train
   '''
    print('Processing Section 3.2')
    if os.path.isfile('a1_3.2.csv'):
        os.remove('a1_3.2.csv')

    train_size_list = [1000, 5000, 10000, 15000, 20000]
    total_list = list(range(len(X_train)))
    accuracy_list = []
    for train_size in train_size_list:
        sample_indice = np.random.choice(total_list, train_size, replace=False)
        X_sample = X_train[sample_indice]
        y_sample = y_train[sample_indice]
        print("training set size: {}".format(train_size))
        accuracy_value, result = classify_func(X_sample, X_test, y_sample, y_test, iBest)
        accuracy_list.append(accuracy_value)
        if train_size == 1000:
            X_1k = X_sample
            y_1k = y_sample

    with open('a1_3.2.csv', 'w',newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        csvwriter.writerow(accuracy_list)
        csvwriter.writerow(["As training set's size goes up, the accuracy on testing set goes up and then tends to be constant."
                            "This is because the model can be characterized better when more data are provided for the training set. "
                            "Meanwhile, the accurancy of testing set will saturate when tranining set reaches a certain size, "
                            "since the model is already characterized well by the set and can not be benefited by the increase of training set."])
    return (X_1k, y_1k)


def process_feature_extract(X_train, y_train, X_test, y_test, iBest, k):
    selector = SelectKBest(f_classif, k)
    X_new = selector.fit_transform(X_train, y_train)
    k_indices = selector.get_support(indices=True)
    print("Test with training set size {}, k = {}".format(len(X_train), k))
    accuracy_value, result = classify_func(X_train[:,k_indices], X_test[:,k_indices], y_train, y_test, iBest)
    pp = selector.pvalues_
    pvalues = pp[k_indices]
    print("{} optimal features are from feature {} (starting at 1)".format(k, k_indices+1))
    return accuracy_value, k_indices, pvalues


def class33(X_train, X_test, y_train, y_test, i, X_1k, y_1k):
    ''' This function performs experiment 3.3
    
    Parameters:
       X_train: NumPy array, with the selected training features
       X_test: NumPy array, with the selected testing features
       y_train: NumPy array, with the selected training classes
       y_test: NumPy array, with the selected testing classes
       i: int, the index of the supposed best classifier (from task 3.1)  
       X_1k: numPy array, just 1K rows of X_train (from task 3.2)
       y_1k: numPy array, just 1K rows of y_train (from task 3.2)
    '''
    print('Processing Section 3.3')
    if os.path.isfile('a1_3.3.csv'):
        os.remove('a1_3.3.csv')

    k_list = [5, 10 , 20, 30, 40 , 50]
    accuracy_list_1k = []
    accuracy_list_32k = []
    features_indices_1k = []
    features_indices_32k = []
    pvalues_list_1k = []
    pvalues_list_32k = []

    for k in k_list:
        accuracy_value, features_indices, pvalues = process_feature_extract(X_1k, y_1k, X_test, y_test, i, k)
        accuracy_list_1k.append(accuracy_value)
        features_indices_1k.append(features_indices)
        pvalues_list_1k.append(pvalues)

        accuracy_value, features_indices, pvalues = process_feature_extract(X_train, y_train, X_test, y_test, i, k)
        accuracy_list_32k.append(accuracy_value)
        features_indices_32k.append(features_indices)
        pvalues_list_32k.append(pvalues)

    with open('a1_3.3.csv', 'w',newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for i in range(6):
            csvwriter.writerow([k_list[i]] + pvalues_list_32k[i].tolist())
        csvwriter.writerow([accuracy_list_1k[0]] + [accuracy_list_32k[0]])
        csvwriter.writerow(["Feature 30, 36, 84, 97 are both chosen at the low(1k) and high(32k) amounts of input data for k=5. "
                            "It is reasonable that the two data set share most of the top features since the smaller one"
                            "is randomly sampled from the larger one so they follow the similar distribution. "
                            "Meanwhile, since the 1k input might be a bit too small, the result from it might not be very robust,"
                            "which could be a reason why there is one out of five features different for the two data set."])
        csvwriter.writerow(["p values are generally lower given more data. This is because usually, "
                            "larger data set is better for characterizing the relationship between the feature and classes, "
                            "thus helping to find the top features to describe the data. "
                            "In that case, the p values will be lower."])
        csvwriter.writerow(["Top 5 features from 32k training case is feature 30, 36, 65, 84, 97. "
                            "They are liwc_AllPunc, liwc_OtherP, liwc_female, liwc_motion and liwc_relativ."
                            "Firstly, it indicates the pre-processing Receptiviti build works better than mine :( "
                            "(All top features are theirs). Then, from the name, we can see things like punctuation,"
                            "motion, other puncutation sounds to be good features to help to classify the catagory."])
    return 0

def class34( filename, i ):
    ''' This function performs experiment 3.4
    
    Parameters
       filename : string, the name of the npz file from Task 2
       i: int, the index of the supposed best classifier (from task 3.1)  
        '''
    print('Processing Section 3.4')
    if os.path.isfile('a1_3.4.csv'):
        os.remove('a1_3.4.csv')

    features = np.load(filename)['arr_0']
    X, y = features[:, :173], features[:, 173]
    results = []


    kf = KFold(n_splits=5, shuffle=True)
    for train_index, test_index in kf.split(X):
        accuracy_per_fold = []
        for i in range(1, 6):
            accuracy_value, result = classify_func(X[train_index], X[test_index], y[train_index], y[test_index], i)
            accuracy_per_fold.append(accuracy_value)
        results.append(accuracy_per_fold)

    accuracy_list = [] # accuracy for each classifier function
    for i in range(5):
        accuracy_list.append([row[i] for row in results])

    Best_class_accuracies = accuracy_list.pop(iBest-1)
    p_value_list = [stats.ttest_rel(Best_class_accuracies,i).pvalue for i in accuracy_list]

    second_Best = 5 if iBest == 4 else 4

    with open('a1_3.4.csv', 'w',newline='') as csvfile:
        csvwriter = csv.writer(csvfile, delimiter=',')
        for i in range(5):
            csvwriter.writerow(results[i])
        csvwriter.writerow(p_value_list)
        csvwriter.writerow(["With 5-fold cross validation, we see the best classifier (classifier {}) from section 3.1 shows a"
                            "significantly better result than other three, but similar to classifier {}. "
                            "With cross validation, all the the examples in the dataset are eventually used for both training and testing, "
                            "which also gives us a 'larger dataset' to use compared with the traditional train/test method."
                            "We still get similar accurancy for classifier 4 & 5, which indicates both classifier perform similarly for the"
                            "given data set, or maybe even with the cross-validation, the data set is still not 'large enough'.".format(iBest, second_Best)])


    return 0
    
if __name__ == "__main__":
    startTime = datetime.now()

    parser = argparse.ArgumentParser(description='Process each .')
    parser.add_argument("-i", "--input", help="the input npz file from Task 2", required=True)
    args = parser.parse_args()

    # TODO : complete each classification experiment, in sequence.
    X_train, X_test, y_train, y_test, iBest = class31(args.input)
    print("Best classifier for this problem is: No.{}".format(iBest))

    (X_1k, y_1k) = class32(X_train, X_test, y_train, y_test, iBest)

    class33(X_train, X_test, y_train, y_test, iBest, X_1k, y_1k)

    class34(args.input, iBest)
    print("Total runtime: {}".format(datetime.now() - startTime))
