from xgboost import XGBClassifier
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

feature_matrix = np.load("music_matrices/music_features_matrix.npy")
genre_truths = np.load("music_matrices/meta_genre_truths.npy")


X_train, X_test, y_train, y_test = train_test_split(feature_matrix, genre_truths, test_size=0.1, random_state=7)

param = {}
param['booster'] = 'gbtree'
param['objective'] = 'binary:logistic'
param["eval_metric"] = "error"
param['eta'] = 0.1
param['gamma'] = 0
param['max_depth'] = 9
param['min_child_weight'] = 1
param['max_delta_step'] = 0
param['subsample']= 0.5
param['colsample_bytree']=1
param['seed'] = 0
param['base_score'] = 0.5

model = XGBClassifier()
model.set_params(**param)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)


accuracy = f1_score(y_test, y_pred, average='weighted')


print("F1 score: %.2f%%" % (accuracy * 100.0))
