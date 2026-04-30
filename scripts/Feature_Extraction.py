import numpy as np
import pandas as pd

"""
Usage:

import Feature_Extraction

matrix = np.load("music_matrices/music_features_matrix.npy")
feature_names = np.load("music_matrices/music_feature_names.npy")
genre_truths = np.load("music_matrices/meta_genre_truths.npy")

contribs = "PCA/feature_contributions.txt"
dims = 15
n_keep = 10

final_matrix, keep_features = Feature_Extraction.extract(matrix, feature_names, genre_truths, dims, n_keep, contribs)

"""

def extract(matrix, feature_names, dims, n_keep, contribs):

	feat_contribs = pd.read_csv(contribs, sep = " ")
	keep_features = []
	
	for i in range(0, dims):
		col_slice = (feat_contribs.iloc[:, i]).sort_values(axis = 0, ascending = False)
		keep_topn = col_slice.head(n_keep)

		keep_features.extend(keep_topn.index.tolist())

	keep_features = set(keep_features)

	new_matrix = pd.DataFrame(matrix)
	new_matrix.columns = feature_names.tolist()

	final_matrix = new_matrix[keep_features]

	return final_matrix.values, keep_features
