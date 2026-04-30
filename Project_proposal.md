# Project Proposal

## Introduction/Background
Musical genres are categorical labels created by humans to characterize pieces of music following a hierarchical structure. The music within one genre is supposed to share more commonalities. The field of Music Genre Classification was originally introduced as a pattern recognition task by Tzanetakis and Cook [1] in 2002. Since then, it has become a widely studied problem in the Music Information Research (MIR) community. Typically, some handcrafted audio features are used as input of a machine learning classifier. Both Supervised learning and unsupervised learning can perform this task.


## Problem Definition
The act of assigning genres to music is commonly performed by humans using pattern recognition. In humans, information from past learnings are used to make inferences about new data that is encountered. However, these classifications can sometimes be opinionated and based off features that are not necessarily characteristic of a certain genre. Using machine learning, we can effectively categorize music based on concrete features used in our dataset and essentially determine the features that impact genre the most.

## Methods
To determine the best method for assigning genres, a series of multiple methods will be implemented and their accuracies will be assessed. Dimensionality reduction by principal component analysis will be used to extract the most meaningful features for labeling. The following methods will be evaluated:

**Unsupervised Methods**:
1. K-Means: While not very flexible, K-Means is fast on large datasets and can serve as a baseline to compare against other clustering methods.
1. Gaussian Mixture Model: GMM has a relatively large amount of flexibility for capturing cluster covariance with respect to K-Means.

**Supervised Methods**:
1. Random Forest Classifier: Random Forest Classifier works well with non-linear, high dimensional data. While we do not yet know the linearity of our dataset, this method would prove advantageous if our dataset is non-linear.
1. Support Vector Machine: SVM is memory efficient and works well with data that can separate with clear margins. While we expect some overlap between similar genres of music to perform poorly under SVM, it’s worth observing whether dissimilar genres are separated more accurately.
1. XGBoost: XGBoost is a flexible decision-tree based method which utilizes non-greedy tree pruning and built in cross validation methods to predict errors with many parameters that can be fine-tuned for optimal clustering.

The following criteria will be assessed to determine which method performed with the highest accuracy:

* Area Under the Curve
* Confusion Matrix
* Precision, Recall, and F1 Scores


## Potential Results
As music can often blend genres, we expect resulting clusters to be noisy, so it is difficult to predict the accuracy of any machine learning method. Soft clustering algorithms could struggle to distinguish between similar genres, while hard clustering algorithms may be inaccurate for multi-genre music samples. Thankfully, our dataset has both single-genre and probabilistic labels from independent contributors. This will allow us to calculate the accuracy of both hard and soft learning algorithms and compare their effectiveness.

## Discussion
Our experiments aim to grade different clustering methods based on their capability in musical classification. We hope to discern optimal algorithms for musical classification given acceptable levels of noise. Further, we might assume a secondary goal of noise reduction- finding high purity and high accuracy clustering methods along closely related genres or sub-genres, i.e. pure sub-genre clustering for Rock'n'Roll.

## References
G. Tzanetakis and P. Cook, "Musical genre classification of audio signals," in IEEE Transactions on Speech and Audio Processing, vol. 10, no. 5, pp. 293-302, July 2002, doi: 10.1109/TSA.2002.800560.

Bogdanov, D., Porter A., Schreiber H., Urbano J., & Oramas S. (2019).
The AcousticBrainz Genre Dataset: Multi-Source, Multi-Level, Multi-Label, and Large-Scale.
20th International Society for Music Information Retrieval Conference (ISMIR 2019).

Oramas, S., Barbieri, F., Nieto, O., & Serra, X. (2018). Multimodal Deep Learning for Music Genre Classification. Transactions of the International Society for Music Information Retrieval, 1(1), 4–21. DOI: http://doi.org/10.5334/tismir.10
