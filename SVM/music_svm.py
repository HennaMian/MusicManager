from sklearn import svm
import sklearn.metrics
import numpy

# parameters for the dataset to use
D = 13
N = 10

# load the feature matrix (NxC)
feature_matrix = numpy.load("./Simplified Folder/music_matrices_reduced_features_top" + str(D) + "_n" + str(N) + "/data_top" + str(D) + "_n" + str(N) + "_clean.npy")

# load the genre truth values (N)
genre_truths = numpy.load("./Simplified Folder/music_matrices_reduced_features_top" + str(D) + "_n" + str(N) + "/labels_top" + str(D) + "_n" + str(N) + "_clean.npy")
#print(genre_truths)

# load the feature names (C)
feature_names = numpy.load("./Simplified Folder/music_matrices_reduced_features_top" + str(D) + "_n" + str(N) + "/music_feature_names_reduced_features_top" + str(D) + "_n" + str(N) + ".npy", allow_pickle=True)
#print(feature_names)

print("loaded data")

# create a SVM classifier
# using a linear kernal because why not
clf = svm.SVC(kernel='linear')
print("created SVM object")

# extract training and test data matrices
N = feature_matrix.shape[0]

test_idx = numpy.random.choice(N, N // 10)

test_matrix = feature_matrix[test_idx, :]
test_genres = genre_truths[test_idx]

train_matrix = numpy.delete(feature_matrix, test_idx, axis=0)
train_genres = numpy.delete(genre_truths, test_idx)

# train the classifier
print("training the SVM object")
clf.fit(train_matrix, train_genres)
print("finished training the SVM object")

print("classifying the test dataset")
predicted = clf.predict(test_matrix)
print("finished classifying the test dataset")

with open("./SVM/prediction.txt", "a") as f:
    f.write("\nD" + str(D) + "N" + str(N) + ">" + str(predicted))

print("determining the model F1 accuracy")
print(sklearn.metrics.f1_score(test_genres, predicted, average="micro"))
