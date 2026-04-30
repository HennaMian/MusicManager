#!/usr/bin/env python3

import csv
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.preprocessing import StandardScaler


# Load data

music_mat = np.load('music_matrices/music_features_matrix.npy')
features = np.load('music_matrices/music_feature_names.npy')

feats = []
for i in features:
	feats.append(i)

np.savetxt("music_matrix.csv", music_mat, delimiter=",")

with open('music_feature_names.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(feats)
