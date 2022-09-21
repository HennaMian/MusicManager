from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

import DataCleaning as dc
import Feature_Extraction

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

##### 3.0 Initiate XGBoost Function #####

def run_XGBoost(matrix, labels, tsize):

	X_train, X_test, y_train, y_test = train_test_split(matrix, labels, test_size=tsize, random_state=7)

	# https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
	param = {}
	param['booster'] = 'gbtree'
	param['objective'] = 'binary:logistic'
	param["eval_metric"] = "error"
	param['eta'] = 0.1 #Analogous to learning rate
	param['gamma'] = 0
	param['max_depth'] = 9
	param['min_child_weight'] = 1
	param['max_delta_step'] = 0
	param['subsample']= 0.5
	param['colsample_bytree']=1
	param['seed'] = 0
	param['base_score'] = 0.5
	#param['alpha'] = 0
	#param['lambda'] = 1

	model = XGBClassifier()
	model.set_params(**param)
	model.fit(X_train, y_train)

	y_pred = model.predict(X_test)

	## Assess F1 Score ##
	# Weighted f1 score here for mutiple labels.
	accuracy = f1_score(y_test, y_pred, average='weighted')

	#print("F1 score: %.2f%%" % (accuracy * 100.0))

	return round(accuracy, 2)

##### 4.0 Train & Evaluate for XGBoost: test_sizes = (0.10, 0.25, 0.50, 0.75, 0.9) #####

log_file = open('xgboost_log.txt', 'w')

test_size_list = [0.10, 0.25, 0.50, 0.75, 0.9]

for ts in test_size_list:
	print("Test Size", ts)
	vals = {} # {dim:{n:[f1, precision, recall]}}

	for d in [7, 10, 13]: # test dims 5-20
		print("Dimensions:", d)
		n_dict = {}
		for n in [3, 10]: # test no. features 1-20
			print("Top feats:", n)
			dims = d
			n_keep = n

			# Feature Extraction
			extracted_matrix, keep_features = Feature_Extraction.extract(cleaned_matrix, feature_names, dims, n_keep, contribs)

			f1 = run_XGBoost(extracted_matrix, new_labels, ts)

			log_file.write("Test Size " + str(ts) + " Dimensions: " + str(d) + " Top feats: " + str(n) + " F1: " + str(f1) + "\n")



