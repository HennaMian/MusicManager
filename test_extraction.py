from DataCleaning import *
import Feature_Extraction

# import the raw data and genres
raw_data = np.load("./music_matrices/music_features_matrix.npy")
raw_genres = np.load("./music_matrices/meta_genre_truths.npy")
raw_features = np.load("./music_matrices/music_feature_names.npy")


### Trial 1: Extract then Clean ###
print("Trial 1: Extract then Clean")

# extract the top 10 n 10 features
extracted_data, extracted_features = Feature_Extraction.extract(raw_data, raw_features, 10, 10, "./PCA/PCA/feature_contributions.txt")
print("extracted data shape", extracted_data.shape)

# clean the extracted data
cleaner = DataCleaner(extracted_data, raw_genres)
cleaned_genres, cleaned_songs = cleaner.CleanDataFilterIn(cleaner)
print("cleaned data shape", cleaned_songs.shape)


### Trial 2: Clean then Extract ###
print()
print("Trial 2: Clean then Extract")

# clean the raw data
cleaner = DataCleaner(raw_data, raw_genres)
cleaned_genres, cleaned_data = cleaner.CleanDataFilterIn(cleaner)
print("cleaned data shape", cleaned_data.shape)

# extract the top 10 n features form the clean data
extracted_data, extracted_features = Feature_Extraction.extract(cleaned_data, raw_features, 10, 10, "./PCA/PCA/feature_contributions.txt")
print("extracted data shape", extracted_data.shape)