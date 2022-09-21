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

dim_list = [7, 10, 13]
feats_list = [3, 10]

# Get input files and output filename
"""
for dim in dim_list:
	for feat in feats_list:
		# load matrix
		mat_path = str("./Simplified Folder/music_matrices_reduced_features_top" + str(dim) + "_n" + str(feat) + \
			"/music_features_matrix_reduced_features_top" + str(dim) + "_n" + str(feat) + ".npy")
		matrix = np.load(mat_path)

		# load feature names
		feat_path = str("./Simplified Folder/music_matrices_reduced_features_top" + str(dim) + "_n" + str(feat) + \
			"/music_feature_names_reduced_features" + str(dim) + "_n" + str(feat) + ".npy")
		feature_names = np.load(feat_path, allow_pickle=True)

		print("feature_names")
		print(feature_names)
		print(feature_names.shape)

		# load genre truths




"""
matrix = np.load("./Simplified Folder/music_matrices_reduced_features_top10_n10/data_top10_n10_clean.npy")
print("Matrix")
print(matrix)
print(matrix.shape)

"""
feature_names = np.load("../Simplified Folder/music_matrices_reduced_features_top10_n10/music_feature_names_reduced_features_top10_n10.npy", allow_pickle=True)

print("feature_names")
print(feature_names)
print(feature_names.shape)


genre_truths = np.load("../Simplified Folder/music_matrices_reduced_features_top10_n10/labels_top10_n10_clean.npy")

print("genre_truths")
print(genre_truths)
print(genre_truths.shape)

unsure = np.load("../Simplified Folder/music_matrices_reduced_features_top10_n10/data_top10_n10_clean.npy")

print("unsure")
print(unsure)
print(unsure.shape)
"""



