from DataCleaning import *
import Feature_Extraction

raw_data = np.load("./music_matrices/music_features_matrix.npy")
raw_genres = np.load("./music_matrices/meta_genre_truths.npy")
raw_features = np.load("./music_matrices/music_feature_names.npy")

def extract_and_clean_data(D, N):
    extracted_data, extracted_features = Feature_Extraction.extract(raw_data, raw_features, D, N, "./PCA/PCA/feature_contributions.txt")

    cleaner = DataCleaner(extracted_data, raw_genres)
    cleaned_genres, cleaned_data = cleaner.CleanDataFilterIn(cleaner)

    with open("./Simplified Folder/music_matrices_reduced_features_top" + str(D) + "_n" + str(N) + "/data_top" + str(D) + "_n" + str(N) + "_clean.npy", "wb") as f:
        np.save(f, cleaned_data)

    with open("./Simplified Folder/music_matrices_reduced_features_top" + str(D) + "_n" + str(N) + "/labels_top" + str(D) + "_n" + str(N) + "_clean.npy", "wb") as f:
        np.save(f, cleaned_genres)

    with open("./Simplified Folder/music_matrices_reduced_features_top" + str(D) + "_n" + str(N) + "/music_feature_names_reduced_features_top" + str(D) + "_n" + str(N) + ".npy", "wb") as f:
        np.save(f, extracted_features)

    with open("./Simplified Folder/music_matrices_reduced_features_top" + str(D) + "_n" + str(N) + "/music_features_matrix_reduced_features_top" + str(D) + "_n" + str(N) + ".npy", "wb") as f:
        np.save(f, extracted_data)

    print("extracted and cleaned the data: ", D, "dims and", N, "features per dim")
    return


D = 7
N = 3
extract_and_clean_data(D, N)

D = 7
N = 10
extract_and_clean_data(D, N)

D = 10
N = 3
extract_and_clean_data(D, N)

D = 10
N = 10
extract_and_clean_data(D, N)

D = 13
N = 3
extract_and_clean_data(D, N)

D = 13
N = 10
extract_and_clean_data(D, N)
