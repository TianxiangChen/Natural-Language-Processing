from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
import numpy as np
from datetime import datetime


def baseline_model():
	# create model
	model = Sequential()
	model.add(Dense(25, input_dim=173, activation='relu'))
	model.add(Dense(4, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model

def classify_func(X, y):
    estimator = KerasClassifier(build_fn=baseline_model, epochs=200, batch_size=5, verbose=0)
    kfold = KFold(n_splits=5, shuffle=True)

    results = cross_val_score(estimator, X, y, cv=kfold)
    print("Baseline: %.2f%% (%.2f%%)" % (results.mean() * 100, results.std() * 100))

    return 0


def classify_tf(filename):
    ''' This function performs classification on the dataset using tensorflow

    Parameters
       filename : string, the name of the npz file from Task 2

    Returns:
        None
    '''
    print('Processing starts')
    features = np.load(filename)['arr_0']

    labels = np.zeros((len(features),4))
    for i in range(len(features)):
        labels[i][int(features[i][173])]= 1
    # X_train, X_test, y_train, y_test = train_test_split(features[:, :173], labels, test_size=0.2)
    classify_func(features[:, :173], labels)
    return 0



if __name__ == "__main__":
    startTime = datetime.now()
    classify_tf('feats.npz')
    print("Total runtime: {}".format(datetime.now() - startTime))
