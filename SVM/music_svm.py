from sklearn import svm
import sklearn.metrics
import numpy

D = 13
N = 10

feature_matrix = numpy.load("./Simplified Folder/music_matrices_reduced_features_top" + str(D) + "_n" + str(N) + "/data_top" + str(D) + "_n" + str(N) + "_clean.npy")

genre_truths = numpy.load("./Simplified Folder/music_matrices_reduced_features_top" + str(D) + "_n" + str(N) + "/labels_top" + str(D) + "_n" + str(N) + "_clean.npy")

feature_names = numpy.load("./Simplified Folder/music_matrices_reduced_features_top" + str(D) + "_n" + str(N) + "/music_feature_names_reduced_features_top" + str(D) + "_n" + str(N) + ".npy", allow_pickle=True)

print("loaded data")

clf = svm.SVC(kernel='linear')
print("created SVM object")

N = feature_matrix.shape[0]

test_idx = numpy.random.choice(N, N // 10)

test_matrix = feature_matrix[test_idx, :]
test_genres = genre_truths[test_idx]

train_matrix = numpy.delete(feature_matrix, test_idx, axis=0)
train_genres = numpy.delete(genre_truths, test_idx)

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
