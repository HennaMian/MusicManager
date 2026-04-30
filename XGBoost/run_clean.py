import numpy as np
from scripts import DataCleaning


labels = np.load('XGBoost/music_matrices/meta_genre_truths.npy')

data = np.load("XGBoost/music_matrices/music_features_matrix.npy")

new_simplified_labels, new_data = DataCleaning()._CleanDataReturnTopN(labels, data)
