import numpy as np
import sklearn as sk
import sklearn.mixture as mixture
from scipy import stats
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import csv
import DataCleaning
import Feature_Extraction
import pandas as pd

class Classifier(object):

    def __init__(self, _path = "", iter_paths=False):
        self.dataMatrix = None
        self.pca_dataMatrix = None
        self.data_Labels = None
        self.unique_data_Labels = None
        self.GMM = None
        self.str = _path
        self.iter = iter_paths
        return

    def Load_Data(self):
        # self.pca_dataMatrix = np.genfromtxt('./music_matrices/music_matrix.csv', delimiter=',')
        # self.pca_dataMatrix = np.genfromtxt('./PCA/PCA/music_matrix.csv', delimiter=',')
        if self.iter:
            self.pca_dataMatrix = np.load('./Simplified Folder/music_matrices_reduced_features_' + self.str + '/data_' + self.str + '_clean.npy')
        # self.data_Labels = np.load('./music_matrices/meta_genre_truths.npy')
        if self.iter:
            self.data_Labels = np.load('./Simplified Folder/music_matrices_reduced_features_' + self.str + '/labels_' + self.str + '_clean.npy')
        self.unique_data_Labels = np.unique(self.data_Labels.flatten())
        return

    # begin region Getters
    def GetData(self):
        return self.pca_dataMatrix

    def GetLabels(self):
        return self.data_Labels

    def GetUniqueLabels(self):
        return self.unique_data_Labels

    def GetGMM(self):
        return self.GMM

    # end region Getters

    # begin region external set
    def SetData(self, new_labels, new_arr):
        self.pca_dataMatrix = new_arr
        self.data_Labels = new_labels
        self.unique_data_Labels = np.unique(self.data_Labels.flatten())
    #

    def _GMM(self, feature_Arr, _cov = 'full'):
        # Return if labels or feature_Arr is empty/not ndarray
        if feature_Arr is None or not isinstance(feature_Arr, np.ndarray):
            return
        self.GMM = mixture.GaussianMixture(n_components=self.unique_data_Labels.shape[0], covariance_type=_cov, max_iter=50000, init_params='random')
        # self.GMM.predict(feature_Arr)

def show_Graph(arr):
    covariances = ['full', 'tied', 'diag', 'spherical']
    display_arr = arr
    plt.figure(figsize=(8, 6))
    fig, axs = plt.subplots(2, 2)
    fig.suptitle(
        "Accuracy of Label Predictions Adjusted for Chance versus Trail (" + str(
            10) + "% withheld for testing) for Musical Classification Using GMM Varying On Covariance Type Per Model")
    for i in range(len(covariances)):
        clusters = [x for x in range(display_arr.shape[0])]
        _adjustedScores = [_gmmARS[i] for _gmmARS in display_arr]
        axs[i % 2, int(i / 2)].title.set_text('Adj Rand Score vs Trial, Cov: ' + str(covariances[i]))
        axs[i % 2, int(i / 2)].plot(clusters, _adjustedScores, '-o', label="Adjusted Random Scores")
        #plt.sca(axs[i % 2, int(i / 2)])
        #plt.xticks(range(len(selected_feats)), ["" for x in range(len(selected_feats))], rotation=30, ha='right')
        if i % 2 == 1:
            #plt.sca(axs[i % 2, int(i / 2)])
            #plt.xticks(range(len(selected_feats)), selected_feats, rotation=30, ha='right')
            axs[i % 2, int(i / 2)].set_xlabel("Trial")
        if int(i / 2) == 0:
            axs[i % 2, int(i / 2)].set_ylabel("Score [Best: 1.0]")
    plt.show()

o_dataCleaner = True
cleaning = False
if cleaning:
    new_Classifier = Classifier()
    new_Classifier.Load_Data()
    new_Cleaner = DataCleaning.DataCleaner()
    new_Classifier.Load_Data()
    #data = new_Cleaner._CleanDataReturnTopN(new_Classifier.GetLabels(), new_Classifier.pca_dataMatrix, 33)
    data = new_Cleaner.CleanDataFilterIn(new_Classifier)
    #print(data)

nothing_Crazy = True
exper = True
donezo = True
if not donezo:
    if exper:
        # flags for testing
        exp = True
        old = False
        stored = True
        _save = False
        _display = True
        loading = False
        # results
        covariances = ['full', 'tied', 'diag', 'spherical']
        selected_feats = ['top7_n3', 'top7_n10', 'top10_n3', 'top10_n10', 'top13_n3', 'top13_n10']
        if not stored and not old:
            feature_names = np.load("./music_matrices/music_feature_names.npy")
            genre_labels = np.load("./music_matrices/meta_genre_truths.npy")
            contribs = "./PCA/PCA/feature_contributions.txt"
            matrix = np.load("./music_matrices/music_features_matrix.npy")
        aic_results = list()
        bic_results = list()
        adjusted_scores = list()
        running_predictions = list()
        running_data = list()
        dim_feat = list()
        # feat/dim
        n_dimensions = 14
        n_features = 33
        # experiment and test results
        if exp:
            if old:
                for _o in range(2, 11):
                    new_Classifier = Classifier()
                    new_Classifier.Load_Data()
                    new_Cleaner = DataCleaning.DataCleaner()
                    new_labels, new_data = new_Cleaner.CleanDataReturnTopN(new_Classifier, _o)
                    new_Classifier.SetData(new_labels, new_data)
                    _newRowAIC = list()
                    _newRowBIC = list()
                    _newRowARS = list()
                    _newRowPred = list()
                    _newRowDat = list()
                    # fit and predict
                    for i in range(len(covariances)):
                        new_Classifier._GMM(new_Classifier.GetData(), covariances[i])
                        gmm = new_Classifier.GetGMM()
                        gmm.fit(new_Classifier.GetData())
                        predictions = gmm.predict(new_Classifier.GetData())
                        _newRowAIC.append(gmm.aic(new_Classifier.GetData()))
                        _newRowBIC.append(gmm.bic(new_Classifier.GetData()))
                        _newRowARS.append(sk.metrics.adjusted_rand_score(new_Classifier.GetLabels().flatten(), predictions.flatten()))
                        _newRowDat.append(new_Classifier.GetData())
                        _newRowPred.append(predictions)
                    _newRowAIC.append(_o)
                    _newRowBIC.append(_o)
                    # check accuracy
                    aic_results.append(_newRowAIC)
                    bic_results.append(_newRowBIC)
                    adjusted_scores.append(_newRowARS)
                    running_data.append(_newRowDat)
                    running_predictions.append(_newRowPred)
            elif stored:
                for p in range(5):
                    adjusted_scores = list()
                    for n_path in range(len(selected_feats)):
                        print('alive ' + selected_feats[n_path])
                        new_Classifier = Classifier(_path=selected_feats[n_path], iter_paths=True)
                        new_Classifier.Load_Data()
                        o_labels, o_data = DataCleaning.DataCleaner().GenreReduction(new_Classifier, 3)
                        new_Classifier.SetData(o_labels, o_data)
                        o_labels, o_data = DataCleaning.DataCleaner().NormalizeGenres(new_Classifier)#GenreReduction(new_Classifier, 3)
                        new_Classifier.SetData(o_labels, o_data)
                        _newRowARS = list()
                        # test set/test labels as necessary
                        if p > 0:
                            labels_train, labels_test, data_train, data_test = sk.model_selection.train_test_split(new_Classifier.GetLabels(), new_Classifier.GetData(), train_size=.1*p, random_state=29)
                        # fit and predict
                        for i in range(len(covariances)):
                            new_Classifier._GMM(new_Classifier.GetData(), covariances[i])
                            gmm = new_Classifier.GetGMM()
                            if p == 0:
                                gmm.fit(new_Classifier.GetData())
                                predictions = gmm.predict(new_Classifier.GetData())
                                _newRowARS.append(sk.metrics.adjusted_rand_score(new_Classifier.GetLabels().flatten(), predictions.flatten()))
                            else:
                                gmm.fit(data_train)
                                predictions = gmm.predict(data_test)
                                _newRowARS.append(sk.metrics.adjusted_rand_score(labels_test.flatten(), predictions.flatten()))

                        adjusted_scores.append(_newRowARS)
                    show_Graph(np.array(adjusted_scores))
                    with open("results_ARS_GMM_"+str(10*p)+"pTesting_teamDataSet.npy", "wb") as f:
                        np.save(f, np.array(adjusted_scores))
            else:
                _o = 0
                for dim in range(6, n_dimensions):
                    for feat in range(2, n_features, 2):
                        print('alive dims ' + str(dim) + ' and feats ' + str(feat))
                        new_Classifier = Classifier(iter_paths=False)
                        out_data, out_features = Feature_Extraction.extract(matrix, feature_names, dim, feat, contribs)
                        out_data = out_data.to_numpy()
                        cleaner = DataCleaning.DataCleaner(out_data, genre_labels)
                        new_labels, new_data = cleaner.CleanDataFilterIn(cleaner)
                        new_Classifier.SetData(new_labels, new_data)
                        _newRowARS = list()
                        # fit and predict
                        labels_train, labels_test, data_train, data_test = sk.model_selection.train_test_split(new_labels, new_data, train_size=.4, random_state=29)
                        for i in range(len(covariances)):
                            new_Classifier._GMM(new_Classifier.GetData(), covariances[i])
                            gmm = new_Classifier.GetGMM()
                            gmm.fit(data_train)
                            predictions = gmm.predict(data_test)
                            _newRowARS.append(
                                sk.metrics.adjusted_rand_score(labels_test.flatten(), predictions.flatten()))
                        adjusted_scores.append(_newRowARS)
                        dim_feat.append([dim, feat, _newRowARS])
                        _o+=1
        elif not exp and not loading:
            new_Classifier.Load_Data()
            new_labels, new_data = new_Cleaner.CleanDataReturnTopN(new_Classifier, 12)
            new_Classifier.SetData(new_labels, new_data)
            new_Classifier._GMM(new_Classifier.GetData())

            # fit and predict
            gmm = new_Classifier.GetGMM()
            gmm.fit(new_Classifier.GetData())
            predictions = gmm.predict(new_Classifier.GetData())
            unique_Cluster, count_Cluster = np.unique(predictions, return_counts=True)
            unique_Genre, count_Genre = np.unique(new_Classifier.GetLabels(), return_counts=True)
            genres = dict()
            for i in range(len(unique_Genre)):
                genres[unique_Genre[i]] = count_Genre[i]
            # sort unique words by count
            sorted_genres = sorted(genres, key=genres.get, reverse=True)
            genre_modes = list()
            remove = list()
            for i in range(len(sorted_genres)):
                _idx = np.where(new_labels == sorted_genres[i])
                _idx = np.take(predictions, _idx)
                _shadow = _idx
                for j in range(len(remove)):
                    _shadow = _shadow[_shadow != remove[j]]
                if len(_shadow) != 0:
                    _idx = _shadow
                mode = stats.mode(_idx, axis=None)[0][0]
                genre_modes.append((sorted_genres[i], genres[sorted_genres[i]], mode))
                remove.append(mode)
            # genre, true count, predicted label
            gmm_actual = dict()
            gmm_predicted = dict()
            gmm_predictions = dict()
            for i in range(len(genre_modes)):
                gmm_predicted[genre_modes[i][0]] = genre_modes[i][2]
                gmm_actual[genre_modes[i][0]] = genre_modes[i][1]
                gmm_predictions[genre_modes[i][0]] = 0
            for i in range(len(genre_modes)):
                # grab indices of genre block
                _idx = np.where(new_labels == genre_modes[i][0])
                val = genre_modes[i][2]
                # grab slice of genre block
                _slice = np.take(predictions, _idx)
                _slice = _slice[_slice != val]
                gmm_predictions[genre_modes[i][0]] = _slice.shape[0]
            ratios = gmm_predictions
            for i in range(len(genre_modes)):
                ratios[genre_modes[i][0]] = abs(ratios[genre_modes[i][0]] - genre_modes[i][1]) / genre_modes[i][1]
                # updating data/labels
                #np.save('Simplified Folder/simplified_labels', new_labels)
                #np.savetxt('Simplified Folder/simplified_music_matrix.csv', new_data, delimiter=',')

            score = sk.metrics.rand_score(new_labels.flatten(), predictions.flatten())
            adj_score = sk.metrics.adjusted_rand_score(new_labels.flatten(), predictions.flatten())
            _aic = gmm.aic(new_data)
            ax = plt.axes(projection='3d')
            ax.scatter3D(new_data[:, 0], new_data[:, 1], new_data[0,2], color="green")
            plt.title('GMM Distribution on 3 features')
            plt.axis('tight')
            plt.show()

        if _display:
            #with open("results_ARS_GMM.npy", "wb") as f:
            #    np.save(f, np.array(dim_feat))
            plt.figure(figsize=(8, 6))
            fig, axs = plt.subplots(2, 2)
            fig.suptitle("Figure 2. Accuracy of Label Predictions Adjusted for Chance versus Trail (Denoted By Extracted Dimensions and Features) for Musical Classification Using GMM Varying On Covariance Type Per Model")

            for i in range(len(covariances)):
                clusters = [x for x in range(len(adjusted_scores))]
                _adjustedScores = [_gmmARS[i] for _gmmARS in adjusted_scores]
                axs[i % 2, int(i/2)].title.set_text('Adj Rand Score vs Trial, Cov: ' + str(covariances[i]))
                axs[i % 2, int(i/2)].plot(clusters, _adjustedScores, '-o', label="Adjusted Random Scores")
                if i % 2 == 1:
                    axs[i % 2, int(i/2)].set_xlabel("Trial Number")
                if int(i/2) == 0:
                    axs[i % 2, int(i/2)].set_ylabel("Score [Best: 1.0]")
                axs[i % 2, int(i/2)].legend(loc='lower right')
            plt.show()
        if loading:
            if _save:
                with open("results_ARS_GMM_60pTraining.npy", "wb") as f:
                    np.save(f, np.array(dim_feat))
            if not _save:
                display_arr = np.load("./results_ARS_GMM_60pTraining.npy", allow_pickle=True)
            else:
                display_arr = np.array(dim_feat)
            #print(display_arr)
            plt.figure(figsize=(8, 6))
            fig, axs = plt.subplots(2, 2)
            fig.suptitle(
                "Figure 2. Accuracy of Label Predictions Adjusted for Chance versus Trail (Denoted By Extracted Dimensions and Features) for Musical Classification Using GMM Varying On Covariance Type Per Model")

            for i in range(len(covariances)):
                clusters = [x for x in range(display_arr.shape[0])]
                _adjustedScores = [_gmmARS[i] for _gmmARS in display_arr[:, 2]]
                axs[i % 2, int(i / 2)].title.set_text('Adj Rand Score vs Trial, Cov: ' + str(covariances[i]))
                axs[i % 2, int(i / 2)].plot(clusters, _adjustedScores, '-o', label="Adjusted Random Scores")
                if i % 2 == 1:
                    axs[i % 2, int(i / 2)].set_xlabel("Trial Number")
                if int(i / 2) == 0:
                    axs[i % 2, int(i / 2)].set_ylabel("Score [Best: 1.0]")
                axs[i % 2, int(i / 2)].legend(loc='lower right')
            plt.show()
            '''
            fig = plt.figure(figsize=(16, 12))
            #fig, axs = plt.subplots(len(running_predictions), 4)
            fig.suptitle("Accuracy of Label Predictions Adjusted for Chance versus Number of Genres Introduced for Musical Classification Using GMM Varying On Covariance Type Per Model")
            for i in range(len(running_predictions)):
                d_data = running_data[i]
                d_predictions = running_predictions[i]
                for j in range(len(covariances)):
                    c_data = d_data[j]
                    c_predictions = d_predictions[j]
                    #axs[i, j].
                    ax = fig.add_subplot(i+1, j+1, 1, projection='3d')
                    ax.scatter3D(c_data[:, 0], c_data[:, 1], c_data[:, 2], c=c_predictions)
            plt.show()
            '''
else:
    if not nothing_Crazy:
        covariances = ['full', 'tied', 'diag', 'spherical']
        selected_feats = ['top7_n3', 'top7_n10', 'top10_n3', 'top10_n10', 'top13_n3', 'top13_n10']
        fig_num = 7  # 2
        stability = True
        if not stability:
            for p in range(5):
                display_arr = np.load("results_ARS_GMM_" + str(10 * p) + "pTesting_teamDataSet.npy")
                plt.figure(figsize=(8, 6))
                fig, axs = plt.subplots(2, 2)
                fig.suptitle(
                    "Figure "+str(fig_num)+". Accuracy of Label Predictions Adjusted for Chance versus Trail ("+str(10*p)+"% withheld for testing) for Musical Classification Using GMM Varying On Covariance Type Per Model")
                for i in range(len(covariances)):
                    clusters = [x for x in range(display_arr.shape[0])]
                    _adjustedScores = [_gmmARS[i] for _gmmARS in display_arr]
                    axs[i % 2, int(i / 2)].title.set_text('Adj Rand Score vs Trial, Cov: ' + str(covariances[i]))
                    axs[i % 2, int(i / 2)].plot(clusters, _adjustedScores, '-o', label="Adjusted Random Scores")
                    plt.sca(axs[i % 2, int(i / 2)])
                    plt.xticks(range(len(selected_feats)), ["" for x in range(len(selected_feats))], rotation=30, ha='right')
                    if i % 2 == 1:
                        plt.sca(axs[i % 2, int(i / 2)])
                        plt.xticks(range(len(selected_feats)), selected_feats, rotation=30, ha='right')
                        axs[i % 2, int(i / 2)].set_xlabel("Trial")
                    if int(i / 2) == 0:
                        axs[i % 2, int(i / 2)].set_ylabel("Score [Best: 1.0]")
                plt.show()
                fig_num+=1
        else:
            n_dimensions = 14
            n_features = 33
            labels = list()
            for i in range(6,n_dimensions):
                for j in range(2, n_features,2):
                    labels.append("top" + str(i) + "_n" + str(j))
            for p in range(5):
                if p == 0:
                    display_arr = np.load("./GMM/results_ARS_GMM.npy", allow_pickle=True)
                else:
                    display_arr = np.load("./GMM/results_ARS_GMM_" + str(100 - 10 * p) + "pTraining.npy", allow_pickle=True)
                plt.figure(figsize=(8, 6))
                fig, axs = plt.subplots(2, 2)
                fig.suptitle(
                    "Figure "+str(fig_num)+". Stability Analysis: Accuracy of Label Predictions Adjusted for Chance versus Trail ("+str(10*p)+"% withheld for testing) for Musical Classification Using GMM Varying On Covariance Type Per Model")
                for i in range(len(covariances)):
                    clusters = [x for x in range(display_arr.shape[0])]
                    _adjustedScores = [_gmmARS[i] for _gmmARS in display_arr[:, 2]]
                    axs[i % 2, int(i / 2)].title.set_text('Adj Rand Score vs Trial, Cov: ' + str(covariances[i]))
                    axs[i % 2, int(i / 2)].plot(clusters, _adjustedScores, '-o', label="Adjusted Random Scores")
                    if i % 2 == 1:
                        axs[i % 2, int(i / 2)].set_xlabel("Trial")
                    if int(i / 2) == 0:
                        axs[i % 2, int(i / 2)].set_ylabel("Score [Best: 1.0]")
                plt.show()
                fig_num+=1


data_set = np.load("./GMM/results_ARS_GMM_10pTesting_teamDataSet.npy", allow_pickle=True)

#accuracies = data_set[:, 2]
#n = np.vstack((accuracies[0], accuracies[1]))
#for i in range(2,accuracies.shape[0]):
#    n = np.vstack((n, accuracies[i]))
print(np.var(data_set, axis=0))
print(np.mean(data_set, axis=0))
show_Graph(data_set)