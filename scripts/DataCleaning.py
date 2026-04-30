import numpy as np
import re
import copy
from collections import Counter

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

    def CleanDataFilterIn(self, new_Classifier, only_top_genres=False, n=10):
        unique_labels = new_Classifier.GetUniqueLabels()

        label_map = {}
        for label in unique_labels:
            label = label.lower().replace(" ", "")

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

        raw_labels = new_Classifier.GetLabels()
        cleaned_labels = [""] * len(raw_labels)

        for i in range(len(raw_labels)):
            label = raw_labels[i].lower().replace(" ", "")
            if label not in label_map:
                cleaned_labels[i] = "ignore"

            else:
                cleaned_labels[i] = label_map[label]

        song_data = new_Classifier.GetData()       

        cleaned_labels = np.array(cleaned_labels)
        song_removal_idx = np.where(cleaned_labels == "ignore")
        cleaned_labels = np.delete(cleaned_labels, song_removal_idx, None)
        cleaned_songs = np.delete(song_data, song_removal_idx, 0)

        label_counter = Counter(cleaned_labels)
        top_labels = label_counter.most_common(n)
        return cleaned_labels, cleaned_songs

    def CleanDataReturnTopN(self, new_Classifier, n=10):
        labels = new_Classifier.GetUniqueLabels()
        labels = ''.join(labels)
        labels = re.sub(r'[^\w]', ' ', labels)
        words = np.array([x.lower() if isinstance(x, str) else x for x in labels.split(sep=' ')])
        words = words[words != ""]
        words = words[words != "all"]
        words = words[words != "the"]
        words = words[words != "male"]
        words = words[words != "female"]
        words = words[words != "age"]
        words = words[words != "misc"]
        unique_words, counts = np.unique(words, return_counts=True)
        word_dict = dict()
        for i in range(len(unique_words)):
            word_dict[unique_words[i]] = counts[i]
        sorted_words = sorted(word_dict, key=word_dict.get, reverse=True)
        _topGenres = list()
        for i in range(n):
            _topGenres.append(sorted_words[i])
        simplified_Labels = new_Classifier.GetLabels()
        for i in range(len(simplified_Labels)):
            for j in range(len(sorted_words)):
                word = sorted_words[j]
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
        new_simplified_labels = np.take(simplified_Labels, indices[0], axis=0).reshape((-1, 1))
        data = new_Classifier.GetData()
        new_data = np.take(data, indices[0], axis=0).reshape((-1, data.shape[1]))
        for i in range(1, len(_topGenres)):
            _labels = np.take(simplified_Labels, indices[i], axis=0).reshape((-1, 1))
            new_simplified_labels = np.concatenate((new_simplified_labels, _labels))
            _data = np.take(data, indices[i], axis=0).reshape((-1, data.shape[1]))
            new_data = np.concatenate((new_data, _data))
        return new_simplified_labels, new_data

    def _CleanDataReturnTopN(self, labels, data, n=10):
        m_labels = np.unique(labels).flatten()
        m_labels = ''.join(m_labels)
        m_labels = re.sub(r'[^\w]', ' ', m_labels)
        words = np.array([x.lower() if isinstance(x, str) else x for x in m_labels.split(sep=' ')])
        words = words[words != ""]
        words = words[words != " "]

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

        unique_words, counts = np.unique(words, return_counts=True)
        word_dict = dict()
        for i in range(len(unique_words)):
            word_dict[unique_words[i]] = counts[i]
        sorted_words = sorted(word_dict, key=word_dict.get, reverse=True)
        _topGenres = list()
        if len(sorted_words) < n:
            n = len(sorted_words)
        for i in range(n):
            _topGenres.append(sorted_words[i])
        simplified_Labels = labels
        for i in range(len(simplified_Labels)):
            for j in range(len(sorted_words)):
                word = sorted_words[j]
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
        new_simplified_labels = np.take(simplified_Labels, indices[0], axis=0).reshape((-1, 1))
        m_data = data
        new_data = np.take(m_data, indices[0], axis=0).reshape((-1, data.shape[1]))
        for i in range(1, len(_topGenres)):
            _labels = np.take(simplified_Labels, indices[i], axis=0).reshape((-1, 1))
            new_simplified_labels = np.concatenate((new_simplified_labels, _labels))
            _data = np.take(m_data, indices[i], axis=0).reshape((-1, data.shape[1]))
            new_data = np.concatenate((new_data, _data))
        unique_words, counts = np.unique(new_simplified_labels, return_counts=True)
        word_dict = dict()
        for i in range(len(unique_words)):
            word_dict[unique_words[i]] = counts[i]
        sorted_words = sorted(word_dict, key=word_dict.get, reverse=True)
        _topGenres = list()
        if len(sorted_words) < n:
            n = len(sorted_words)
        for i in range(n):
            _topGenres.append(sorted_words[i])
        return new_simplified_labels, new_data

    def GenreReduction(self, new_Classifier, n=10):
        words = new_Classifier.GetLabels()
        unique_words, counts = np.unique(words, return_counts=True)
        word_dict = dict()
        for i in range(len(unique_words)):
            word_dict[unique_words[i]] = counts[i]
        sorted_words = sorted(word_dict, key=word_dict.get, reverse=True)
        _topGenres = list()
        for i in range(n):
            _topGenres.append(sorted_words[i])
        simplified_Labels = new_Classifier.GetLabels()
        indices = list()
        for i in range(len(_topGenres)):
            _indices = np.where(simplified_Labels == _topGenres[i])
            indices.append(_indices)
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
        words = new_Classifier.GetLabels()
        unique_words, counts = np.unique(words, return_counts=True)
        word_dict = dict()
        for i in range(unique_words.shape[0]):
            word_dict[unique_words[i]] = counts[i]
        sorted_words = sorted(word_dict, key=word_dict.get, reverse=True)
        _topGenres = list()
        for i in range(unique_words.shape[0]):
            _topGenres.append(sorted_words[i])
        min_count = word_dict[_topGenres[-1]]
        simplified_Labels = new_Classifier.GetLabels()
        indices = list()
        for i in range(len(_topGenres)):
            _indices = np.where(simplified_Labels == _topGenres[i])
            indices.append(_indices[0][:min_count])
        new_simplified_labels = np.take(simplified_Labels, indices[0], axis=0).reshape((-1, 1))
        data = new_Classifier.GetData()
        new_data = np.take(data, indices[0], axis=0).reshape((-1, data.shape[1]))
        for i in range(1, len(_topGenres)):
            _labels = np.take(simplified_Labels, indices[i], axis=0).reshape((-1, 1))
            new_simplified_labels = np.concatenate((new_simplified_labels, _labels))
            _data = np.take(data, indices[i], axis=0).reshape((-1, data.shape[1]))
            new_data = np.concatenate((new_data, _data))
        return new_simplified_labels, new_data
