import numpy

# N: number of songs
# C: number of classes (features)


# load the feature matrix (NxC)
feature_matrix = numpy.load("./music_features_matrix.npy")

# load the genre truth values (N)
genre_truths = numpy.load("./meta_genre_truths.npy")

# load the recording IDs for the songs (N)
recording_ids = numpy.load("./meta_recording_ids.npy")

# load the recording titles (N)
music_titles = numpy.load("./meta_music_titles.npy")

# load the feature names (C)
feature_names = numpy.load("./music_feature_names.npy")
