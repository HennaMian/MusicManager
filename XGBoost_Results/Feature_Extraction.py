import numpy as np
import pandas as pd

"""
Usage:

import Feature_Extraction

# Get input files and output filename

matrix = np.load("/Dataset/music_features_matrix.npy")
feature_names = np.load("/Dataset/music_feature_names.npy")
genre_truths = np.load("/Dataset/meta_genre_truths.npy")

### Feature Extraction ###

contribs = "/Dataset/PCA/feature_contributions.txt"
dims = 15 # change this number to however many dimensions to keep
n_keep = 10 # change this numner to however many top features per dimensions to keep

final_matrix, keep_features = Feature_Extraction.extract(matrix, feature_names, genre_truths, dims, n_keep, contribs)

"""

### Feature Extraction ###

def extract(matrix, feature_names, dims, n_keep, contribs):

	# 1.0 Extract features to keep
	feat_contribs = pd.read_csv(contribs, sep = " ")
	keep_features = []
	
	for i in range(0, dims):
		col_slice = (feat_contribs.iloc[:, i]).sort_values(axis = 0, ascending = False)
		keep_topn = col_slice.head(n_keep)

		keep_features.extend(keep_topn.index.tolist())

	keep_features = set(keep_features)

	# 2.0 Subset matrix 
	new_matrix = pd.DataFrame(matrix)
	new_matrix.columns = feature_names.tolist()

	final_matrix = new_matrix[keep_features]

	return final_matrix, keep_features


