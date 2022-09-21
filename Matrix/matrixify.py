
# VARIABLES - CHANGE THESE TO MODIFY THE OUTPUT MATRIX

num_songs = 15000  # number of songs to gather (gathers in recursive directory order)
datapath = "../Datasets/acousticbrainz-mediaeval-validation/"  # relative path of the datasets (this path works if you run this script in the Matrix folder)

####################


import json
import os
import numpy


# check if a data item is a singular value: string, int, or float
def check_singular(item):
    if isinstance(item, str) or isinstance(item, int) or isinstance(item, float):
        return True
    return False


# pretty print a list
def print_list(l):
    for i in l:
        print("-", i)
    return


# get all subfiled recursively
def get_all_subfiles():

    #we shall store all the file names in this list
    filelist = []

    for root, dirs, files in os.walk(datapath):
        for file in files:
            #append the file name to the list
            filelist.append(os.path.join(root,file))

    return filelist


# from a given json path, get the feature names
def get_feature_names(path):
    feature_names = []

    with open(path, "r") as f:
        data = json.load(f)

        for key, val in data.items():
            for subkey, subval in val.items():
                if isinstance(subval, str) or isinstance(subval, int) or isinstance(subval, float):
                    feature = subkey
                    value = subval
                    feature_names.append([key, feature, value, ""])
                        
                if isinstance(subval, dict) and "median" in subval and check_singular(subval["median"]):
                    feature = subkey
                    value = subval["median"]
                    feature_names.append([key, feature, value, "m"])
    return feature_names


# convert a note to a float
def convert_note_to_float(note):
    mapping = {
        "A": 1,
        "A#": 1.5,
        "B": 2,
        "B#": 2.5,
        "C": 3,
        "C#": 3.5,
        "D": 4,
        "D#": 4.5,
        "E": 5,
        "E#": 5.5,
        "F": 6,
        "F#": 6.5,
        "G": 7,
        "G#": 7.5,

        "minor": 0,
        "major": 1,
    }
    return mapping[note]


# look at the music to generate a matrix of features
def generate_feature_matrix():
    datapath = "../Datasets/acousticbrainz-mediaeval-validation/28/"

    paths = get_all_subfiles()
    num_paths = len(paths)

    feature_names = get_feature_names(paths[0])
    feature_matrix = numpy.zeros((num_songs, len(feature_names)))

    genre_truths = [""] * num_songs
    titles = [""] * num_songs
    recording_ids = [""] * num_songs
    
    song_id = 0
    for fname in paths:
        # break when we have enough songs (we will disregard badly formed tracked)
        if song_id >= num_songs:
            break

        feature_count = 0

        with open(fname, "r") as f:
            data = json.load(f)

            # first get the track's metadata
            if "genre" not in data["metadata"]["tags"]:
                continue
            genre_truth = data["metadata"]["tags"]["genre"][0]

            
            if "title" not in data["metadata"]["tags"]: # ignore songs with titles that don't exist
                continue

            title = data["metadata"]["tags"]["title"][0]
            recording_id = data["metadata"]["tags"]["musicbrainz_recordingid"][0]

            # store the metadata as Python arrays (not numpy, yet)
            genre_truths[song_id] = genre_truth
            titles[song_id] = title
            recording_ids[song_id] = recording_id

            # then fill in the feature matrix row for this track
            col = 0
            for feature in feature_names:
                val = data[feature[0]][feature[1]]
                if isinstance(val, str):
                    val = convert_note_to_float(val)

                if feature[3] == "m":
                    # if the median does not exist, put 0
                    if "median" not in val:
                        print(val)
                        feature_matrix[song_id, col] = 0
                    else:
                        feature_matrix[song_id, col] = val["median"]
                else:
                    feature_matrix[song_id, col] = val
                col += 1       

        song_id += 1
        print("song", song_id, "/", num_paths)

    # save the feature array to a file
    with open("music_features_matrix.npy", "wb") as f:
        numpy.save(f, feature_matrix)

    # save the feature names to a file
    with open("music_feature_names.npy", "wb") as f:
        numpy.save(f, numpy.array([x[1] for x in feature_names]))
    
    # save the ground genre truths to a file
    with open("meta_genre_truths.npy", "wb") as f:
        numpy.save(f, numpy.array(genre_truths))
    
    # save the music titles to a file
    with open("meta_music_titles.npy", "wb") as f:
        numpy.save(f, numpy.array(titles))

    # save the recording IDs to a file
    with open("meta_recording_ids.npy", "wb") as f:
        numpy.save(f, numpy.array(recording_ids))

    print("Generated and saved music and meta matrices!")

generate_feature_matrix()