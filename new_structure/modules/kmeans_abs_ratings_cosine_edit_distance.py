import math

import gensim
import nltk
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

from new_structure.modules.datastructs.metaphor import Metaphor
from new_structure.modules.datastructs.metaphor_group import MetaphorGroup


def create_word2vec_model():
    # Cosine Similarity model creation
    model = KeyedVectors.load_word2vec_format(
        './modules/datastructs/wiki.en.vec')

    return model


def get_abstractness_rating():
    csv = pd.read_csv(
        "./data/AC_ratings_google3m_koeper_SiW.csv",
        error_bad_lines=False)
    csv = pd.DataFrame(csv)
    csv_df = pd.DataFrame(csv['WORD\tRATING'].str.split('\t', 1).tolist(),
                          columns=['word', 'rating'])
    abstractness_rating_dict = pd.Series(csv_df.rating.values, index=csv_df.word).to_dict()
    return abstractness_rating_dict


abstractness_rating_dict = get_abstractness_rating()


def get_cosine_similarity_model(df):
    data = []
    for j in zip(df.adj, df.noun):
        temp = [j[0], j[1]]
        data.append(temp)
    model = gensim.models.Word2Vec(data, min_count=1, size=200, window=5)
    return model


def vectorize_data(df):
    an_vectorized = []

    model = get_cosine_similarity_model(df)

    for j in zip(df.adj, df.noun):
        a = j[0]
        n = j[1]
        l = []
        if '-' in a:
            s = a.split('-')
            ar_Adj = (float(abstractness_rating_dict[s[0]]) + float(abstractness_rating_dict[s[1]])) / 2
        else:
            ar_Adj = float(abstractness_rating_dict[a])
        if '-' in n:
            s = n.split('-')
            ar_Noun = (float(abstractness_rating_dict[s[0]]) + float(abstractness_rating_dict[s[1]])) / 2
        else:
            ar_Noun = float(abstractness_rating_dict[n])
        l.append((ar_Adj) / 10)
        l.append((ar_Noun) / 10)
        l.append((np.sign(ar_Adj - ar_Noun)))
        s = model.similarity(j[0], j[1])
        l.append(s)
        l.append(nltk.edit_distance(j[0], j[1]) / 10)

        an_vectorized.append(list(l))

    an_vectorized = np.asarray(an_vectorized)
    return an_vectorized


def vectorize_data_abstractness(df):
    an_vectorized = []

    model = get_cosine_similarity_model(df)

    abstractness_rating_dict = get_abstractness_rating()

    for j in zip(df.adj, df.noun):
        a = j[0]
        n = j[1]
        l = []
        if '-' in a:
            s = a.split('-')
            ar_Adj = (float(abstractness_rating_dict[s[0]]) + float(abstractness_rating_dict[s[1]])) / 2
        else:
            ar_Adj = float(abstractness_rating_dict[a])
        if '-' in n:
            s = n.split('-')
            ar_Noun = (float(abstractness_rating_dict[s[0]]) + float(abstractness_rating_dict[s[1]])) / 2
        else:
            ar_Noun = float(abstractness_rating_dict[n])
        l.append((ar_Adj) / 10)
        l.append((ar_Noun) / 10)
        l.append((np.sign(ar_Adj - ar_Noun)))

        an_vectorized.append(list(l))

    an_vectorized = np.asarray(an_vectorized)
    return an_vectorized


def vectorize_data_abstractness_cosine(df):
    an_vectorized = []

    model = get_cosine_similarity_model(df)

    abstractness_rating_dict = get_abstractness_rating()

    for j in zip(df.adj, df.noun):
        a = j[0]
        n = j[1]
        l = []
        if '-' in a:
            s = a.split('-')
            ar_Adj = (float(abstractness_rating_dict[s[0]]) + float(abstractness_rating_dict[s[1]])) / 2
        else:
            ar_Adj = float(abstractness_rating_dict[a])
        if '-' in n:
            s = n.split('-')
            ar_Noun = (float(abstractness_rating_dict[s[0]]) + float(abstractness_rating_dict[s[1]])) / 2
        else:
            ar_Noun = float(abstractness_rating_dict[n])
        l.append((ar_Adj) / 10)
        l.append((ar_Noun) / 10)
        l.append((np.sign(ar_Adj - ar_Noun)))
        s = model.similarity(j[0], j[1])
        l.append(s)

        an_vectorized.append(list(l))

    an_vectorized = np.asarray(an_vectorized)
    return an_vectorized


def get_adj_noun_class(df, adjective, noun):
    row = df.loc[(df['adj'] == adjective) & (df['noun'] == noun)]
    adj_noun_class = row["class"].tolist()
    if adj_noun_class:
        return adj_noun_class[0]


def k_mean_distance(data, cx, cy, i_centroid, cluster_labels):
    distances = [np.sqrt((x - cx) ** 2 + (y - cy) ** 2) for (x, y) in data[cluster_labels == i_centroid]]
    return distances


def k_mean_1d_distance(data, cx, i_centroid, cluster_labels):
    distances = [(x - cx) for x in data[cluster_labels == i_centroid]]
    return distances


def find_standard_deviation(an_vectorized, clustering_labels, test_data_coordinates, predicted_data_labels):
    test_train_coordinates = np.concatenate((an_vectorized, test_data_coordinates), axis=0)
    test_train_labels = np.concatenate((clustering_labels, predicted_data_labels), axis=0)

    cluster_map = pd.DataFrame()
    cluster_map['coordinatesx'] = test_train_coordinates[:, 0]
    cluster_map['coordinatesy'] = test_train_coordinates[:, 1]
    cluster_map['label'] = test_train_labels

    cluster1_coordinates = cluster_map[cluster_map.label == 1][['coordinatesx', 'coordinatesy']]
    cluster2_coordinates = cluster_map[cluster_map.label == 0][['coordinatesx', 'coordinatesy']]

    cluster_1_standard_deviation = np.var(cluster1_coordinates, axis=0)
    cluster_2_standard_deviation = np.var(cluster2_coordinates, axis=0)

    cluster_1_std = math.sqrt(
        cluster_1_standard_deviation['coordinatesx'] ** 2 + cluster_1_standard_deviation['coordinatesy'] ** 2)
    cluster_2_std = math.sqrt(
        cluster_2_standard_deviation['coordinatesx'] ** 2 + cluster_2_standard_deviation['coordinatesy'] ** 2)

    return cluster_1_std, cluster_2_std


def cross_validation(train_df, test_df, kmeans_clustering):
    scores = []
    cv = KFold(n_splits=10, random_state=42, shuffle=False)
    X = train_df.iloc[:, [0, 1]]
    y = train_df.iloc[:, 2]
    for train_index, test_index in cv.split(train_df):
        print("Train Index: ", type(train_index), train_index.tolist(), "\n")
        print("Test Index: ", test_index.tolist())

        train_index = train_index.tolist()
        test_index = test_index.tolist()
        print("x index={}  y index={}".format(X.index, y.index))

        X_train = X[train_index]
        X_test = X[test_index]
        y_train = y[train_index]
        y_test = y[test_index]
        #     best_svr.fit(X_train, y_train)
        #     Fitting the model after each iteration of cv
        kmeans_clustering.fit(X_train, y_train)

        scores.append(kmeans_clustering.score(X_test, y_test))


def get_confidence(an_vectorized, kmeans_clustering, test_data_coordinates, predicted_data_labels):
    centroids = kmeans_clustering.cluster_centers_
    clustering_labels = kmeans_clustering.labels_

    confidence_dict = {}
    X_dist = kmeans_clustering.transform(an_vectorized) ** 2

    centroid_list = centroids.tolist()
    distances = []

    if centroids.size <= 2:
        for i, cx in enumerate(centroids):
            mean_distance = k_mean_1d_distance(an_vectorized, cx, i, clustering_labels)

            distances.append(mean_distance)
    else:
        for i, (cx, cy) in enumerate(centroids):
            mean_distance = k_mean_distance(an_vectorized, cx, cy, i, clustering_labels)
            # mean_distance = k_mean_distance(an_vectorized_PCA, centroid_list[0][i], centroid_list[1][i], i, clusters)
            distances.append(mean_distance)

    max_indices = []
    for label in np.unique(kmeans_clustering.labels_):
        X_label_indices = np.where(clustering_labels == label)[0]
        max_label_idx = X_label_indices[np.argmax(X_dist[clustering_labels == label].sum(axis=1))]
        max_indices.append(max_label_idx)

    test_data_coordinate_list = test_data_coordinates.tolist()
    predicted_label_list = predicted_data_labels.tolist()
    for i in range(len(test_data_coordinate_list)):
        if predicted_label_list[i] == 0:
            data_point_center_distance = distance.euclidean(centroid_list[0], test_data_coordinate_list[i])

            center_distance_other_cluster = distance.euclidean(centroid_list[1], test_data_coordinate_list[i])

            confidence_dict[i] = 1 - (
                    (data_point_center_distance) / (data_point_center_distance + center_distance_other_cluster))

        elif predicted_label_list[i] == 1:
            data_point_center_distance = distance.euclidean(centroid_list[1], test_data_coordinate_list[i])
            center_distance_other_cluster = distance.euclidean(centroid_list[0], test_data_coordinate_list[i])

            confidence_dict[i] = 1 - (
                    (data_point_center_distance) / (data_point_center_distance + center_distance_other_cluster))

    return confidence_dict


accuracy_list = []


def identify_metaphors_abstractness_cosine_edit_dist(candidates, cand_type, verbose: str) -> MetaphorGroup:
    results = MetaphorGroup()
    candidates_list = candidates.candidates
    if not candidates_list:
        return results
    components = 2
    if len(candidates_list) < 2:
        components = len(candidates_list)
    MET_AN_EN = pd.read_table(
        './data/training_adj_noun_met_en.txt',
        delim_whitespace=True, names=('adj', 'noun'))
    MET_AN_EN['class'] = 1
    LIT_AN_EN = pd.read_table(
        './data/training_adj_noun_nonmet_en.txt',
        delim_whitespace=True, names=('adj', 'noun'))
    LIT_AN_EN['class'] = 0

    fields = ['adj', 'noun']
    MET_AN_EN_TEST = pd.read_excel(
        './data/Datasets_ACL2014.xlsx',
        sheetname='MET_AN_EN', usecols=fields)
    MET_AN_EN_TEST['class'] = 1
    LIT_AN_EN_TEST = pd.read_excel(
        './data/Datasets_ACL2014.xlsx',
        sheetname='LIT_AN_EN', usecols=fields)
    LIT_AN_EN_TEST['class'] = 0

    df = pd.concat([LIT_AN_EN, MET_AN_EN, MET_AN_EN_TEST, LIT_AN_EN_TEST])

    data_list = []

    for c in candidates:
        dataframe_data = {}
        dataframe_data["adj"] = c.getSource()
        dataframe_data["noun"] = c.getTarget()
        dataframe_data["class"] = get_adj_noun_class(df, c.getSource(), c.getTarget())
        if not isinstance(dataframe_data["class"], (bool, int)):
            dataframe_data["class"] = 2
        data_list.append(dataframe_data)
    df_test_data = pd.DataFrame.from_records(data_list)

    user_input_df = pd.concat([df_test_data], axis=0).reset_index()

    an_vectorized = vectorize_data(df)
    an_vectorized_user_input = vectorize_data(user_input_df)

    an_vectorized_training_PCA = PCA(n_components=components).fit_transform(an_vectorized)
    an_vectorized_test_PCA = PCA(n_components=components).fit_transform(an_vectorized_user_input)
    kmeans_clustering = KMeans(n_clusters=2, random_state=43)
    idx = kmeans_clustering.fit(an_vectorized_training_PCA)
    y1 = idx.predict(an_vectorized_test_PCA)

    confidence = get_confidence(an_vectorized_training_PCA, kmeans_clustering, an_vectorized_test_PCA, y1)

    print("Confidence of the corresponding words are : {} ".format(confidence))
    accuracy_list.append(accuracy_score(np.asarray(user_input_df['class']), y1))
    print('Accuracy is: ', accuracy_list)
    user_input_df['predict'] = y1
    confidence_counter = -1
    for c in candidates:
        confidence_counter += 1
        adj = c.getSource()
        noun = c.getTarget()
        candidate_df = user_input_df.loc[(user_input_df['adj'] == adj) & (user_input_df['noun'] == noun)]
        if len(candidate_df.index):
            result_class = candidate_df.iloc[0]['predict']
            if result_class.any() == 0:
                result = False
            else:
                result = True

            results.addMetaphor(Metaphor(c, result, confidence[confidence_counter]))
    return results
