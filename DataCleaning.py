import numpy as np
import re
import copy
from collections import Counter

#-----------------------------------------------------------------------------#
#    You're now entering the most poorly written data cleaning script ever    #
#-----------------------------------------------------------------------------#



class DataCleaner(object):
    def __init__(self, _data=None, _labels=None):
        self.data = _data
        self.labels = _labels
        return

    def GetData(self):
        return self.data
    def GetUniqueLabels(self):
        return np.unique(self.labels)
    def GetLabels(self):
        return self.labels

    # This method will filter in genres that we can use
    def CleanDataFilterIn(self, new_Classifier, only_top_genres=False, n=10):
        
        # get the unique labels
        unique_labels = new_Classifier.GetUniqueLabels()
        

        # create a map from raw labels to cleaned labels
        label_map = {}
        
        # for each label, filter it if possible
        for label in unique_labels:
            # make the label lowercase
            #print(label)
            label = label.lower().replace(" ", "")

            # combine common labels
            if "rock" in label:
                label_map[label] = "rock"
            elif "afri" in label:
                label_map[label] = "african"
            elif "telegu" in label or "india" in label or "bollywood" in label or "hindi" in label:
                label_map[label] = "indian"
            elif "brazilian" in label or "hispanic" in label or "salsa" in label or "samba" in label or "mexican" in label or "mariachi" in label or "cuban" in label or "puerto" in label or "latin" in label or "merengue" in label:
                label_map[label] = "hispanic"
            elif "celtic" in label or "galicia" in label:
                label_map[label] = "celtic"
            elif "folk" in label:
                label_map[label] = "folk"
            elif "christian" in label or "gospel" in label or "church" in label or "praise" in label or "religious" in label:
                label_map[label] = "gospel"
            elif "hip" in label or "hop" in label or "r&b" in label or "randb" in label or "rap" in label:
                label_map[label] = "hiphop"
            elif "classical" in label or "ballet" in label or "medieval" in label or "baroque" in label or "chamber" in label or "orches" in label or "fugue" in label or "sonata" in label or "gregorian" in label:
                label_map[label] = "classical"
            elif "afro" in label or "reggae" in label or "jama" in label:
                label_map[label] = "afro"
            elif "chill" in label or "rave" in label or "break" in label or "edm" in label or "trap" in label or "electro" in label or "dubstep" in label or "house" in label or "dance" in label or "tech" in label or "trance" in label or "acid" in label:
                label_map[label] = "electronic"
            elif "blues" in label or "soul" in label or "wop" in label:
                label_map[label] = "blues"
            elif "jazz" in label or "bop" in label or "swing" in label:
                label_map[label] = "jazz"
            elif "country" in label or "bluegrass" in label:
                label_map[label] = "country"
            elif "brutal" in label or "metal" in label or "death" in label or "doom" in label or "thrash" in label or "grime" in label:
                label_map[label] = "metal"
            elif "punk" in label or "emo" in label or "goth" in label or "scream" in label:
                label_map[label] = "punk"
            elif "funk" in label or "groov" in label or "motown" in label:
                label_map[label] = "funk"
            elif "disco" in label:
                label_map[label] = "disco"
            elif "pop" in label or "top40" in label:
                label_map[label] = "pop"

                
            # DEBUG: otherwise, classify the label as itself -- remove this when we have enough good genres
            #else:
                #print("Currently unclassified label:", label)
                #label_map[label] = label

        # get the raw labels and instantiate the cleaned label
        raw_labels = new_Classifier.GetLabels()
        cleaned_labels = [""] * len(raw_labels)
        
        
        # for each song, use the label map to set its label
        for i in range(len(raw_labels)):
            label = raw_labels[i].lower().replace(" ", "")
            # if the label is not mapped, ignore that song
            if label not in label_map:
                #print("Label is not mapped:", label)
                cleaned_labels[i] = "ignore"

            # otherwise use the map to group the label
            else:
                cleaned_labels[i] = label_map[label]

        # get the song data for song removal
        song_data = new_Classifier.GetData()       
        
        # filter out songs that have an "ignore" label
        cleaned_labels = np.array(cleaned_labels)
        song_removal_idx = np.where(cleaned_labels == "ignore")
        cleaned_labels = np.delete(cleaned_labels, song_removal_idx, None)
        cleaned_songs = np.delete(song_data, song_removal_idx, 0)

        # get the top labels
        label_counter = Counter(cleaned_labels)
        top_labels = label_counter.most_common(n)
        #print("Top labels:", top_labels)
        # return the resulting labels and songs
        return cleaned_labels, cleaned_songs
        


    # This method will update classifier with new labels and corresponding data for top n genres
    def CleanDataReturnTopN(self, new_Classifier, n=10):
        # grab a string to process
        labels = new_Classifier.GetUniqueLabels()
        # join our strings
        labels = ''.join(labels)
        # remove any non-alphanumeric elements
        labels = re.sub(r'[^\w]', ' ', labels)
        # back to numpy land
        words = np.array([x.lower() if isinstance(x, str) else x for x in labels.split(sep=' ')])
        # remove empty strings
        words = words[words != ""]
        # remove nonsense/garbage
        words = words[words != "all"]
        words = words[words != "the"]
        words = words[words != "male"]
        words = words[words != "female"]
        words = words[words != "age"]
        words = words[words != "misc"]
        # get unique words and unique word counts
        unique_words, counts = np.unique(words, return_counts=True)
        # build our dictionary of unique words : unique word count for later
        word_dict = dict()
        for i in range(len(unique_words)):
            word_dict[unique_words[i]] = counts[i]
        # sort unique words by count
        sorted_words = sorted(word_dict, key=word_dict.get, reverse=True)
        # let's just print out our top n and see where we're at
        _topGenres = list()
        for i in range(n):
            #num = word_dict[sorted_words[i]]
            #print(sorted_words[i] + ' ' + str(num))
            _topGenres.append(sorted_words[i])
        # try to clean up some labels
        simplified_Labels = new_Classifier.GetLabels()
        for i in range(len(simplified_Labels)):
            for j in range(len(sorted_words)):
                word = sorted_words[j]
                # word = to_str(word)
                if word in str(simplified_Labels[i]).lower() and len(word) > 2:
                    if "hop" in word or "hip" in word or "힙합" in str(simplified_Labels[i]).lower() or "r&b" in str(
                            simplified_Labels[i]).lower():
                        simplified_Labels[i] = "hip-hop"
                        break
                    if "electro" in word or "tech" in word or "dance" in word or "house" in word:
                        simplified_Labels[i] = "electronic"
                        break
                    simplified_Labels[i] = word
                    break
        indices = list()
        for i in range(len(_topGenres)):
            _indices = np.where(simplified_Labels == _topGenres[i])
            indices.append(_indices)
        # reduce data set
        new_simplified_labels = np.take(simplified_Labels, indices[0], axis=0).reshape((-1, 1))
        data = new_Classifier.GetData()
        new_data = np.take(data, indices[0], axis=0).reshape((-1, data.shape[1]))
        for i in range(1, len(_topGenres)):
            _labels = np.take(simplified_Labels, indices[i], axis=0).reshape((-1, 1))
            new_simplified_labels = np.concatenate((new_simplified_labels, _labels))
            _data = np.take(data, indices[i], axis=0).reshape((-1, data.shape[1]))
            new_data = np.concatenate((new_data, _data))
        return new_simplified_labels, new_data

    # General Use: This method will update classifier with new labels and corresponding data for top n genres
    # labels should be size n x 1 (song genre labels)
    # data should be size n x d (songs by features)
    def _CleanDataReturnTopN(self, labels, data, n=10):
        # grab a string to process
        m_labels = np.unique(labels).flatten()
        # join our strings
        m_labels = ''.join(m_labels)
        # remove any non-alphanumeric elements
        m_labels = re.sub(r'[^\w]', ' ', m_labels)
        # back to numpy land
        words = np.array([x.lower() if isinstance(x, str) else x for x in m_labels.split(sep=' ')])
        # remove empty strings
        words = words[words != ""]
        words = words[words != " "]

        # remove nonsense/garbage
        words = words[words != "all"]
        words = words[words != "the"]
        words = words[words != "male"]
        words = words[words != "female"]
        words = words[words != "age"]
        words = words[words != "misc"]
        words = words[words != "and"]
        words = words[words != "r"]
        words = words[words != "old"]
        words = words[words != "new"]
        words = words[words != "post"]
        words = words[words != "hard"]


        # get unique words and unique word counts
        unique_words, counts = np.unique(words, return_counts=True)
        # build our dictionary of unique words : unique word count for later
        word_dict = dict()
        for i in range(len(unique_words)):
            word_dict[unique_words[i]] = counts[i]
        # sort unique words by count
        sorted_words = sorted(word_dict, key=word_dict.get, reverse=True)
        # let's just print out our top n and see where we're at
        _topGenres = list()
        if len(sorted_words) < n:
            n = len(sorted_words)
        for i in range(n):
            # num = word_dict[sorted_words[i]]
            # print(sorted_words[i] + ' ' + str(num))
            _topGenres.append(sorted_words[i])
        # try to clean up some labels
        simplified_Labels = labels
        for i in range(len(simplified_Labels)):
            for j in range(len(sorted_words)):
                word = sorted_words[j]
                # word = to_str(word)
                if word in str(simplified_Labels[i]).lower() and len(word) > 2:
                    if "hop" in word or "hip" in word or "힙합" in str(simplified_Labels[i]).lower() or "r&b" in str(
                            simplified_Labels[i]).lower():
                        simplified_Labels[i] = "hip-hop"
                        break
                    if "electro" in word or "tech" in word or "dance" in word or "house" in word:
                        simplified_Labels[i] = "electronic"
                        break
                    if "rock" in word:
                        simplified_Labels[i] = "rock"
                        break
                    if "classic" in word:
                        simplified_Labels[i] = "classic"
                        break
                    simplified_Labels[i] = word
                    break
        indices = list()
        for i in range(len(_topGenres)):
            _indices = np.where(simplified_Labels == _topGenres[i])
            indices.append(_indices)
        # reduce data set
        new_simplified_labels = np.take(simplified_Labels, indices[0], axis=0).reshape((-1, 1))
        m_data = data
        new_data = np.take(m_data, indices[0], axis=0).reshape((-1, data.shape[1]))
        for i in range(1, len(_topGenres)):
            _labels = np.take(simplified_Labels, indices[i], axis=0).reshape((-1, 1))
            new_simplified_labels = np.concatenate((new_simplified_labels, _labels))
            _data = np.take(m_data, indices[i], axis=0).reshape((-1, data.shape[1]))
            new_data = np.concatenate((new_data, _data))
        # new_unique, new_count = np.unique(new_simplified_labels, return_counts=True)
        unique_words, counts = np.unique(new_simplified_labels, return_counts=True)
        # build our dictionary of unique words : unique word count for later
        word_dict = dict()
        for i in range(len(unique_words)):
            word_dict[unique_words[i]] = counts[i]
        # sort unique words by count
        sorted_words = sorted(word_dict, key=word_dict.get, reverse=True)
        # let's just print out our top n and see where we're at
        _topGenres = list()
        if len(sorted_words) < n:
            n = len(sorted_words)
        for i in range(n):
            # num = word_dict[sorted_words[i]]
            # print(sorted_words[i] + ' ' + str(num))
            _topGenres.append(sorted_words[i])
        return new_simplified_labels, new_data

        # This method will update classifier with new labels and corresponding data for top n genres
    def GenreReduction(self, new_Classifier, n=10):
        # grab a string to process
        words = new_Classifier.GetLabels()
        # get unique words and unique word counts
        unique_words, counts = np.unique(words, return_counts=True)
        # build our dictionary of unique words : unique word count for later
        word_dict = dict()
        for i in range(len(unique_words)):
            word_dict[unique_words[i]] = counts[i]
        # sort unique words by count
        sorted_words = sorted(word_dict, key=word_dict.get, reverse=True)
        # let's just print out our top n and see where we're at
        _topGenres = list()
        for i in range(n):
            # num = word_dict[sorted_words[i]]
            # print(sorted_words[i] + ' ' + str(num))
            _topGenres.append(sorted_words[i])
        # try to clean up some labels
        simplified_Labels = new_Classifier.GetLabels()
        indices = list()
        for i in range(len(_topGenres)):
            _indices = np.where(simplified_Labels == _topGenres[i])
            indices.append(_indices)
        # reduce data set
        new_simplified_labels = np.take(simplified_Labels, indices[0], axis=0).reshape((-1, 1))
        data = new_Classifier.GetData()
        new_data = np.take(data, indices[0], axis=0).reshape((-1, data.shape[1]))
        for i in range(1, len(_topGenres)):
            _labels = np.take(simplified_Labels, indices[i], axis=0).reshape((-1, 1))
            new_simplified_labels = np.concatenate((new_simplified_labels, _labels))
            _data = np.take(data, indices[i], axis=0).reshape((-1, data.shape[1]))
            new_data = np.concatenate((new_data, _data))
        return new_simplified_labels, new_data

    def NormalizeGenres(self, new_Classifier):
        # grab a string to process
        words = new_Classifier.GetLabels()
        # get unique words and unique word counts
        unique_words, counts = np.unique(words, return_counts=True)
        # build our dictionary of unique words : unique word count for later
        word_dict = dict()
        for i in range(unique_words.shape[0]):
            word_dict[unique_words[i]] = counts[i]
        # sort unique words by count
        sorted_words = sorted(word_dict, key=word_dict.get, reverse=True)
        # let's just print out our top n and see where we're at
        _topGenres = list()
        for i in range(unique_words.shape[0]):
            # num = word_dict[sorted_words[i]]
            # print(sorted_words[i] + ' ' + str(num))
            _topGenres.append(sorted_words[i])
        # grab min
        min_count = word_dict[_topGenres[-1]]
        # try to clean up some labels
        simplified_Labels = new_Classifier.GetLabels()
        indices = list()
        for i in range(len(_topGenres)):
            _indices = np.where(simplified_Labels == _topGenres[i])
            indices.append(_indices[0][:min_count])
        # reduce data set
        new_simplified_labels = np.take(simplified_Labels, indices[0], axis=0).reshape((-1, 1))
        data = new_Classifier.GetData()
        new_data = np.take(data, indices[0], axis=0).reshape((-1, data.shape[1]))
        for i in range(1, len(_topGenres)):
            _labels = np.take(simplified_Labels, indices[i], axis=0).reshape((-1, 1))
            new_simplified_labels = np.concatenate((new_simplified_labels, _labels))
            _data = np.take(data, indices[i], axis=0).reshape((-1, data.shape[1]))
            new_data = np.concatenate((new_data, _data))
        return new_simplified_labels, new_data