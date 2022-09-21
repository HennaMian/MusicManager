#!/Users/maggiebrown/opt/anaconda3/bin/python

import csv
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns; sns.set()
from sklearn.preprocessing import StandardScaler


# Load data

music_mat = np.load('/Users/maggiebrown/ML_Spring2021/Project/New_Data/music_features_matrix.npy')

#labels = np.load('/Users/maggiebrown/ML_Spring2021/Project/Dataset/meta_genre_truths.npy')

features = np.load('/Users/maggiebrown/ML_Spring2021/Project/New_Data/music_feature_names.npy')

feats = []
for i in features:
	feats.append(i)

np.savetxt("music_matrix.csv", music_mat, delimiter=",")

with open('music_feature_names.csv', 'w') as myfile:
    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    wr.writerow(feats)

