from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt

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

log_file = open('emperical_log.txt', 'w')

test_size_list = [0.10, 0.25, 0.50, 0.75, 0.9]

for ts in test_size_list:
	print("Test Size", ts)
	vals = {} # {dim:{n:[f1, precision, recall]}}

	for d in range(5,21,5): # test dims 5-20
		print("Dimensions:", d)
		n_dict = {}
		for n in range(3,21,3): # test no. features 1-20
			print("Top feats:", n)
			dims = d
			n_keep = n

			# Feature Extraction
			extracted_matrix, keep_features = Feature_Extraction.extract(cleaned_matrix, feature_names, dims, n_keep, contribs)

			f1 = run_XGBoost(extracted_matrix, new_labels, ts)

			log_file.write("Test Size " + str(ts) + " Dimensions: " + str(d) + " Top feats: " + str(n) + " F1: " + str(f1) + "\n")

			if d not in vals.keys():
				n_dict[n] = [f1]
				vals[d] = n_dict
			else:
				n_dict = vals[d]
				n_dict[n] = [f1]
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






