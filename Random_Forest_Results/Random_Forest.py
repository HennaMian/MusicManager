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

import DataCleaning as dc
import Feature_Extraction

"""
This Script will:
1. Perform Random Forest
2. Evaluate the ML classifiers's performance including:
	a. precision, recall and F1 scores
"""

### 1.0 Load Data ###

matrix = np.load("./Random_Forest_Results/New_Data/music_features_matrix.npy")
feature_names = np.load("./Random_Forest_Results/New_Data/music_feature_names.npy")
genre_truths = np.load("./Random_Forest_Results/New_Data/meta_genre_truths.npy")

contribs = "./Random_Forest_Results/New_Data/PCA/feature_contributions.txt"

### 2.0 Data Cleaning ###

print("Cleaning Data...")

cleaner = dc.DataCleaner(matrix, genre_truths)
cleaned_out = cleaner.CleanDataFilterIn(cleaner)

new_labels = cleaned_out[0]
cleaned_matrix = cleaned_out[1]

##### 3.0 Initiate RF Function #####

def random_forest(matrix, truths, tsize):

	print("Matrix Size:", extracted_matrix.shape)

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

##### 4.0 Train & Evaluate for RF #####

log_file = open('RF_log.txt', 'w')

test_size_list = [0.10, 0.25, 0.50, 0.75, 0.9]

for ts in test_size_list:
	print("Test Size", ts)
	vals = {} # {dim:{n:[f1, precision, recall]}}

	for d in [7, 10, 13]: # test dims
		print("Dimensions:", d)
		n_dict = {}
		for n in [3, 10]: # test no. features
			print("Top feats:", n)
			dims = d
			n_keep = n

			# Feature Extraction
			extracted_matrix, keep_features = Feature_Extraction.extract(cleaned_matrix, feature_names, dims, n_keep, contribs)

			f1, recall, precision = random_forest(extracted_matrix, new_labels, ts)

			log_file.write("Test Size " + str(ts) + " Dimensions: " + str(d) + " Top feats: " + str(n) + " F1: " + str(f1) + "\n")
			log_file.write("Test Size " + str(ts) + " Dimensions: " + str(d) +  " Top feats: " + str(n) +  " Recall: " + str(recall) + "\n")
			log_file.write("Test Size " + str(ts) + " Dimensions: " + str(d) + " Top feats: " + str(n) + " Precision: " + str(precision) + "\n")











