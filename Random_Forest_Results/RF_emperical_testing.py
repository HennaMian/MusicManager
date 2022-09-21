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
1. Perform Random Forest on different:
	a. dimensions/features
	b. test sizes
2. Evaluate the ML classifiers's performance including:
	a. precision, recall and F1 scores
	b. plot presicion, recall, and F1 scores as dimensions and number of features vary
"""

##### 1.0 Load Data #####

matrix = np.load("./New_Data/music_features_matrix.npy")
feature_names = np.load("./New_Data/music_feature_names.npy")
genre_truths = np.load("./New_Data/meta_genre_truths.npy")

contribs = "./New_Data/PCA/feature_contributions.txt"

##### 2.0 Clean Data #####

cleaner = dc.DataCleaner(matrix, genre_truths)
cleaned_out = cleaner.CleanDataFilterIn(cleaner)

new_labels = cleaned_out[0]
cleaned_matrix = cleaned_out[1]

print("cleaned_matrix", cleaned_matrix.shape)
print("new_labels", new_labels.shape)

##### 3.0 Initiate RF Function #####

log_file = open('emperical_log.txt', 'w')

def random_forest(matrix, truths, tsize):

	print("Matrix Size:", matrix.shape)

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

##### 4.0 Train & Evaluate for RF: test_sizes = (0.10, 0.25, 0.50, 0.75, 0.9) #####

test_size_list = [0.10, 0.25, 0.50, 0.75, 0.9]

for ts in test_size_list:
	print("Test Size", ts)
	vals = {} # {dim:{n:[f1, precision, recall]}}

	for d in range(5,21,5): # test dims 5-20
		print("Dimensions:", d)
		n_dict = {}
		for n in range(3,16,3): # test no. features 1-15
			print("Top feats:", n)
			dims = d
			n_keep = n

			# Feature Extraction
			extracted_matrix, keep_features = Feature_Extraction.extract(cleaned_matrix, feature_names, dims, n_keep, contribs)

			f1, recall, precision = random_forest(extracted_matrix, new_labels, ts)

			log_file.write("Test Size " + str(ts) + " Dimensions: " + str(d) + " Top feats: " + str(n) + " F1: " + str(f1) + "\n")
			log_file.write("Test Size " + str(ts) + " Dimensions: " + str(d) +  " Top feats: " + str(n) +  " Recall: " + str(recall) + "\n")
			log_file.write("Test Size " + str(ts) + " Dimensions: " + str(d) + " Top feats: " + str(n) + " Precision: " + str(precision) + "\n")

			if d not in vals.keys():
				n_dict[n] = [f1, recall, precision]
				vals[d] = n_dict
			else:
				n_dict = vals[d]
				n_dict[n] = [f1, recall, precision]
				vals[d] = n_dict

	## Plotting ##
	# x = number of dimensions to use
	# y = number of features
	# z = score (F1, precision, or recall)

	# plot F1 scores:
	x = [] 
	y = []
	z = []

	for dim in vals.keys():
		for feats in vals[dim].keys():
			score = vals[dim][feats][0]

			x.append(dim)
			y.append(feats)
			z.append(round(score, 2))

	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')

	ax.scatter(x, y, z)
	ax.set_xlabel('Dimensions')
	ax.set_ylabel('No. Features')
	ax.set_zlabel('F1 Score')
	plt.title("Test Size " + str(ts) + " F1 Scores")
	plt.savefig("Emperical_Testing_Plots/Test_Size_" + str(ts) + "/F1_scores.png")

	# max F1 score:
	max_f1 = max(z)
	max_index = z.index(max_f1) 

	newline = str("Test Size " + str(ts) + " Max F1 Score: " + str(max_f1) + "\n")
	log_file.write(newline)

	newline = str("Test Size " + str(ts) + " Max F1 Score Dim: " + str(x[max_index]) + "\n")
	log_file.write(newline)

	newline = str("Test Size " + str(ts) + " Max F1 Score Feaures: " + str(y[max_index]) + "\n")
	log_file.write(newline)

	# plot precision scores:
	x = [] 
	y = []
	z = []

	for dim in vals.keys():
		for feats in vals[dim].keys():
			score = vals[dim][feats][1]

			x.append(dim)
			y.append(feats)
			z.append(round(score, 2))

	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')

	ax.scatter(x, y, z)
	ax.set_xlabel('Dimensions')
	ax.set_ylabel('No. Features')
	ax.set_zlabel('Precision Score')
	plt.title("Test Size " + str(ts) + " Precision Scores")
	plt.savefig("Emperical_Testing_Plots/Test_Size_" + str(ts) + "/precision_scores.png")

	# max precision score:
	max_f1 = max(z)
	max_index = z.index(max_f1) 

	newline = str("Test Size " + str(ts) + " Max Precision Score: " + str(max_f1) + "\n")
	log_file.write(newline)

	newline = str("Test Size " + str(ts) + " Max Precision Score Dim: " + str(x[max_index]) + "\n")
	log_file.write(newline)

	newline = str("Test Size " + str(ts) + " Max Precision Score Feaures: " + str(y[max_index]) + "\n")
	log_file.write(newline)

	# plot recall scores:
	x = [] 
	y = []
	z = []

	for dim in vals.keys():
		for feats in vals[dim].keys():
			score = vals[dim][feats][2]

			x.append(dim)
			y.append(feats)
			z.append(round(score, 2))

	fig = plt.figure()
	ax = fig.add_subplot(projection='3d')

	ax.scatter(x, y, z)
	ax.set_xlabel('Dimensions')
	ax.set_ylabel('No. Features')
	ax.set_zlabel('Recall Score')
	plt.title("Test Size " + str(ts) + " Recall Scores")
	plt.savefig("Emperical_Testing_Plots/Test_Size_" + str(ts) + "/recall_scores.png")

	# max recall score:
	max_f1 = max(z)
	max_index = z.index(max_f1) 

	newline = str("Test Size " + str(ts) + " Max Recall Score: " + str(max_f1) + "\n")
	log_file.write(newline)

	newline = str("Test Size " + str(ts) + " Max Recall Score Dim: " + str(x[max_index]) + "\n")
	log_file.write(newline)

	newline = str("Test Size " + str(ts) + " Max Recall Score Feaures: " + str(y[max_index]) + "\n")
	log_file.write(newline)





