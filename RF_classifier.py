from itertools import cycle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import label_binarize
from sklearn.model_selection import learning_curve
from sklearn.svm import LinearSVC
from sklearn.datasets import make_classification
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import sklearn.metrics as sklearn
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OutputCodeClassifier
from scipy import interp
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import validation_curve
import sys

import DataCleaning
import Feature_Extraction

"""
This Script will:
1. Perform Random Forest
2. Evaluate the ML classifiers's performance including:
	a. precision, recall and F1 scores
	b. receiving operating cures (ROCs) - In Progress
	c. validation curves (test for over/underfitting) - In Progress


To Do:

1. change paths
2. implement data cleaning/feature extraction (may do this in separate script)
3. change RF implementation to function; split data many ways to train/test
4. implement ROCs and validation curves
"""

# Get input files and output filename

matrix = np.load("/Users/maggiebrown/ML_Spring2021/Project/Dataset/music_features_matrix.npy")
feature_names = np.load("/Users/maggiebrown/ML_Spring2021/Project/Dataset/music_feature_names.npy")
genre_truths = np.load("/Users/maggiebrown/ML_Spring2021/Project/Dataset/meta_genre_truths.npy")

### Data Cleaning ###

#cleaned_matrix = DataCleaning.CleanDataReturnTopN(self, new_Classifier, n=10)

### Feature Extraction ###

contribs = "/Users/maggiebrown/ML_Spring2021/Project/Dataset/PCA/feature_contributions.txt"
dims = 13
n_keep = 8

final_matrix, keep_features = Feature_Extraction.extract(matrix, feature_names, genre_truths, dims, n_keep, contribs)

# print(final_matrix.shape, len(keep_features))

##### Initiate RF Function #####

def random_forest(matrix, truths, tsize):

	X = matrix
	y = truths

	print("Training...")

	x_train, x_test, y_train, y_test = train_test_split(X, y , test_size = tsize, shuffle=True, random_state=100)

	print("Modeling...")

	model = OutputCodeClassifier(RandomForestClassifier(n_estimators=100, random_state=0), code_size=2, random_state=0)
	model.fit(x_train, y_train)

	y_pred = model.predict(x_test)

	# F1 score, precision, recall
	recall = sklearn.recall_score(y_test, y_pred, average='macro')
	precision = sklearn.precision_score(y_test, y_pred, average='macro')
	f1 = sklearn.f1_score(y_test, y_pred, average='macro')

	return f1, recall, precision

##### Train & Evaluate for RF: test_size = 0.25 #####

f1, recall, precision = random_forest(final_matrix, genre_truths, 0.25)

print("F1 score =  %s" % (f1))
print("Recall =  %s" % (recall))
print("Precision = %s" % (precision))


