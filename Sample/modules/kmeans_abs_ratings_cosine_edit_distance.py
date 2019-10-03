import csv
import math
import time
import warnings
from collections import Counter

import gensim
import matplotlib
# import matplotlib.pyplot as plt
import nltk
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from scipy.spatial import distance
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.metrics import accuracy_score
from sklearn.metrics.cluster import completeness_score
from sklearn.metrics.cluster import homogeneity_score
from sklearn.metrics.cluster import v_measure_score
from sklearn.model_selection import KFold

from Sample.modules.datastructs.metaphor import Metaphor
from Sample.modules.datastructs.metaphor_group import MetaphorGroup

matplotlib.use('Agg')
from matplotlib import pyplot as plt

# from new_structure.modules.datastructs.metaphor import Metaphor
# from new_structure.modules.datastructs.metaphor_group import MetaphorGroup

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UndefinedMetricWarning)


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


def get_kMeans_fit(components):
    df = get_training_data()
    an_vectorized = vectorize_data(df)
    # components = 2
    # if len(candidates_list) < 2:
    #     components = len(candidates_list)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        an_vectorized_training_PCA = PCA(n_components=components).fit_transform(an_vectorized)
    kMeans_clustering1 = KMeans(n_clusters=2, n_init=1000, n_jobs=-1)
    idx_kMeans_fit = kMeans_clustering1.fit(an_vectorized_training_PCA)
    return idx_kMeans_fit, kMeans_clustering1, an_vectorized_training_PCA


def get_cosine_similarity_model(df):
    data = []
    for j in zip(df.adj, df.noun):
        temp = [j[0], j[1]]
        data.append(temp)
    model = gensim.models.Word2Vec(data, min_count=1, size=200, window=5)
    return model


csv_columns = ['Source', 'Target', 'TrueLabel', 'PredictLabel', 'Confidence', 'Accuracy']
csv_file = "results" + str(int(time.time())) + ".csv"


def create_csv_w_headers():
    try:
        with open(csv_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
            writer.writeheader()
    except IOError:
        print("I/O error")


create_csv_w_headers()


def vectorize_data(df):
    an_vectorized = []
    l = []
    model = get_cosine_similarity_model(df)
    name = 0
    for j in zip(df.adj, df.noun):
        a = j[0]
        n = j[1]
        l = []
        abs = []
        # abs2 = []
        # abs3 = []
        # cosi=[]
        # edit=[]
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

        # abs1.append((ar_Adj) / 10)
        # abs2.append((ar_Noun) / 10)
        # abs3.append((np.sign(ar_Adj - ar_Noun)))
        # cosi.append(model.similarity(j[0], j[1]))
        # edit.append(nltk.edit_distance(j[0], j[1]) / 10)
        # l = [abs1,abs2,abs3, cosi, edit]
        # print(l)

        an_vectorized.append(list(l))

    # an_vectorized = np.asarray(l)
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

    # if centroids.size <= 2:
    #     for i, cx in enumerate(centroids):
    #         mean_distance = k_mean_1d_distance(an_vectorized, cx, i, clustering_labels)
    #
    #         distances.append(mean_distance)
    # else:
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
accuracy_list_wo_class2 = []
word_pairs = []
accuracy_confidence_list = []
accuracy_confidence_list_wo_class2 = []
training_data = None


def get_training_data():
    global training_data
    # global user_data
    try:
        if not training_data.empty:
            return training_data
    except:
        """training data is empty"""
    # try:
    #     if not user_data.empty:
    #         return user_data
    # except:
    #     """user data is empty"""
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
        sheet_name='MET_AN_EN', usecols=fields)
    MET_AN_EN_TEST['class'] = 1
    LIT_AN_EN_TEST = pd.read_excel(
        './data/Datasets_ACL2014.xlsx',
        sheet_name='LIT_AN_EN', usecols=fields)
    LIT_AN_EN_TEST['class'] = 0

    df = pd.concat([LIT_AN_EN, MET_AN_EN, MET_AN_EN_TEST, LIT_AN_EN_TEST])
    # df = pd.concat([LIT_AN_EN, MET_AN_EN])
    # userdf = pd.concat([LIT_AN_EN_TEST,MET_AN_EN_TEST])
    training_data = df
    return df


acc_counter = 0
idx_kMeans_fit = {}


def identify_metaphors_abstractness_cosine_edit_dist(candidates, cand_type, verbose: str) -> MetaphorGroup:
    # cross_validation_acc_presc(an_vectorized,df)
    results = MetaphorGroup()
    candidates_list = candidates.candidates
    if not candidates_list:
        return results
    components = 2
    # if len(candidates_list) < 2:
    #     components = len(candidates_list)

    global idx_kMeans_fit
    # if idx_kMeans_fit.get(components):
    #
    #     idx, kmeans_clustering, an_vectorized_training_PCA = idx_kMeans_fit.get(components)
    # else:
    # idx_kMeans_fit[components] = get_kMeans_fit(components)
    idx, kmeans_clustering, an_vectorized_training_PCA = get_kMeans_fit(components)

    data_list = []
    df = get_training_data()

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
    user_input_df_wo_class2 = user_input_df.loc[user_input_df['class'] != 2]

    # an_vectorized = vectorize_data(df)
    an_vectorized_user_input = vectorize_data(user_input_df)
    an_vectorized_user_input_wo_class2 = vectorize_data(user_input_df_wo_class2)
    # print("Printing an vectorized and uer input")
    # print(an_vectorized)
    # print(an_vectorized_user_input)
    # cross_val_df = an_vectorized
    # print("cross_valdf",cross_val_df.shape())
    # y_cross_val = df['class']

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # an_vectorized_training_PCA = PCA(n_components=components).fit_transform(an_vectorized)
        an_vectorized_test_PCA = PCA(n_components=components).fit_transform(an_vectorized_user_input)
        an_vectorized_test_PCA_wo_class2 = PCA(n_components=components).fit_transform(
            an_vectorized_user_input_wo_class2)
    # an_vectorized_training_PCA = an_vectorized
    # an_vectorized_test_PCA = an_vectorized_user_input
    # kmeans_clustering = KMeans(n_clusters=2,n_init=1000,n_jobs=-1)
    # idx = kmeans_clustering.fit(an_vectorized_training_PCA)
    # y_train = kmeans_clustering.fit_predict(an_vectorized_training_PCA)

    # idx,kmeans_clustering,an_vectorized_training_PCA = idx_kMeans_fit
    # kmeans_clustering = kMeans_clustering
    y1 = idx.predict(an_vectorized_test_PCA)
    y2 = idx.predict(an_vectorized_test_PCA_wo_class2)
    confidence = get_confidence(an_vectorized_training_PCA, kmeans_clustering, an_vectorized_test_PCA, y1)
    # confidence_wo_class2 = get_confidence(an_vectorized_training_PCA, kmeans_clustering, an_vectorized_test_PCA_wo_class2, y2)

    print("Confidence of the corresponding words are : {} ".format(confidence))
    # print("Confidence of the corresponding words are without class2: {} ".format(confidence_wo_class2))
    sentence_accuracy = accuracy_score(np.asarray(user_input_df['class']), y1)
    sentence_accuracy_wo_class2 = accuracy_score(np.asarray(user_input_df_wo_class2['class']), y2)
    accuracy_list.append(sentence_accuracy)
    accuracy_list_wo_class2.append(sentence_accuracy_wo_class2)

    print('Accuracy is: ', accuracy_list)
    print('Accuracy is without class2: ', accuracy_list_wo_class2)
    # if len(accuracy_list) > 0 and accuracy_list[-1] > 0:
    #     global acc_counter
    #     acc_counter = acc_counter + 1
    # print("highest accuracy:", acc_counter)
    # cross_validation_acc_presc(an_vectorized, df)

    user_input_df['predict'] = y1
    # calc_homogenity_comp_vmeas_training(df, y_train)

    calc_homogenity_comp_vmeas(user_input_df, candidates)

    confidence_counter = -1
    for c in candidates:
        results_dict = {}

        confidence_counter += 1
        adj = c.getSource()
        noun = c.getTarget()
        candidate_df = user_input_df.loc[(user_input_df['adj'] == adj) & (user_input_df['noun'] == noun)]
        # print(candidate_df["adj"][confidence_counter])
        # print(candidate_df["noun"][confidence_counter])
        if candidate_df["class"][confidence_counter] != 2:
            word_pairs.append(
                "{}||{}".format(candidate_df["adj"][confidence_counter], candidate_df["noun"][confidence_counter]))
            results_dict["Source"] = candidate_df["adj"][confidence_counter]
            results_dict["Target"] = candidate_df["noun"][confidence_counter]
            results_dict["TrueLabel"] = candidate_df["class"][confidence_counter]
            results_dict["PredictLabel"] = candidate_df["predict"][confidence_counter]
            results_dict["Confidence"] = confidence[confidence_counter]

            print(word_pairs)
            if candidate_df["class"][confidence_counter] == candidate_df["predict"][confidence_counter]:
                accuracy_confidence_list.append([1, confidence[confidence_counter]])
                results_dict["Accuracy"] = 1

            else:
                accuracy_confidence_list.append([0, confidence[confidence_counter]])
                results_dict["Accuracy"] = 0
            try:
                with open(csv_file, 'a') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=csv_columns)
                    # writer.writeheader()
                    writer.writerow(results_dict)
            except IOError:
                print("I/O error")
        if len(candidate_df.index):
            result_class = candidate_df.iloc[0]['predict']
            if result_class.any() == 0:
                result = False
            else:
                result = True

            results.addMetaphor(Metaphor(c, result, confidence[confidence_counter]))

    # plot_accuracy_confidence_histogram(word_pairs, accuracy_confidence_list)
    # plot_each_accuracy_confidence_histogram(word_pairs, accuracy_confidence_list)
    try:
        # plot_histogram(accuracy_confidence_list)
        # plot_accuracy_confidence_ratio_histogram(accuracy_confidence_list)
        # plot_accuracy_confidence_boxplot(accuracy_confidence_list)
        plot_accuracy_percentage_conf_bin(accuracy_confidence_list)
    except Exception as e:
        print("exception during plotting the graph -> {}".format(e))
    # plot_accuracy_confidence(word_pairs, accuracy_confidence_list)

    return results


def cross_validation_acc_presc(an_vectorized, df):
    # New Cross validation

    from sklearn.model_selection import cross_validate
    from sklearn.model_selection import cross_val_score

    # MET_AN_EN = pd.read_table('/home/shruti/metaphor_idenfication/new_structure/data/training_adj_noun_met_en.txt', delim_whitespace=True, names=('adj', 'noun'))
    # MET_AN_EN['class'] = 1
    # LIT_AN_EN = pd.read_table('/home/shruti/metaphor_idenfication/new_structure/data/training_adj_noun_met_en.txt', delim_whitespace=True, names=('adj', 'noun'))
    # LIT_AN_EN['class'] = 0
    #
    # fields = ['adj', 'noun']
    # MET_AN_EN_TEST = pd.read_excel('/home/shruti/metaphor_idenfication/new_structure/data/Datasets_ACL2014.xlsx', sheetname='MET_AN_EN', usecols=fields)
    # MET_AN_EN_TEST['class'] = 1
    # LIT_AN_EN_TEST = pd.read_excel('/home/shruti/metaphor_idenfication/new_structure/data/Datasets_ACL2014.xlsx', sheetname='LIT_AN_EN', usecols=fields)
    # LIT_AN_EN_TEST['class'] = 0
    #
    # df = pd.concat([MET_AN_EN, LIT_AN_EN, LIT_AN_EN_TEST, MET_AN_EN_TEST], ignore_index=True)
    # df = pd.DataFrame(df)
    #
    # # df = pd.concat([LIT_AN_EN,MET_AN_EN] ,ignore_index=True)
    # # df = pd.DataFrame(df)
    #
    # data = []
    # for j in zip(df.adj, df.noun):
    #     temp = [j[0], j[1]]
    #     data.append(temp)
    # model = gensim.models.Word2Vec(data, min_count=1, size=200, window=5)
    # # model = KeyedVectors.load_word2vec_format('wiki.en.vec')
    #
    # an_vectorized = []
    # for j in zip(df.adj, df.noun):
    #     a = j[0]
    #     n = j[1]
    #     l = []
    #     if '-' in a:
    #         s = a.split('-')
    #         ar_Adj = (float(dict[s[0]]) + float(dict[s[1]])) / 2
    #     else:
    #         ar_Adj = float(dict[a])
    #     if '-' in n:
    #         s = n.split('-')
    #         ar_Noun = (float(dict[s[0]]) + float(dict[s[1]])) / 2
    #     else:
    #         ar_Noun = float(dict[n])
    #     l.append((ar_Adj) / 10)
    #     l.append((ar_Noun) / 10)
    #     l.append((np.sign(ar_Adj - ar_Noun)))
    #     l.append(model.similarity(j[0], j[1]))
    #     l.append(nltk.edit_distance(j[0], j[1]) / 10)
    #
    #     # an_vectorized = np.asarray([ [(ar_Adj)/10],[(ar_Noun)/10], [np.sign(ar_Adj - ar_Noun)]])
    #     an_vectorized.append(list(l))
    #     # an_vectorized.append(( ar_Adj + ar_Noun+ np.sign(ar_Adj - ar_Noun))/10)
    # an_vectorized = np.asarray(an_vectorized)

    kmeans_clustering = KMeans(n_clusters=2, n_jobs=-1, n_init=1000)
    X = an_vectorized
    y = df['class']
    # cv = KFold(n_splits=2,random_state=49999999,shuffle=False)

    # cv1 = ShuffleSplit(n_splits=2, random_state=0)
    idx = kmeans_clustering.fit(X, y)
    #     Ytrain = idx.predict(X_train)
    # Ytest = idx.predict(X_test)

    scoring = ['precision_macro', 'recall_macro']
    cross_val_scores = cross_val_score(kmeans_clustering, X, y, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (cross_val_scores.mean(), cross_val_scores.std() * 2))
    print("Crossvalscores:", cross_val_scores)

    scores = cross_validate(kmeans_clustering, X, y, scoring=scoring,
                            cv=10)
    print("Scores Keys:", sorted(scores.keys()))

    # print("Scores:train_precision_macro",scores['train_precision_macro'])

    # print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
    print("Scores:")
    for key, value in scores.items():
        print(key)
        print(value)
        print("Avg score: %0.2f (+/- %0.2f)" % (value.mean(), value.std() * 2))


def plot_accuracy_confidence(x, y):
    plt.xlabel("X-axis")
    plt.ylabel("Y-axis")
    plt.title("Accuracy Confidence Comparison")
    plt.figure(figsize=(41, 11))
    labels = {0: "Accuracy", 1: "Confidence"}
    if x and y:
        for i in range(len(y[0])):
            plt.plot(x, [pt[i] for pt in y], label=labels[i])
        plt.legend()
        print("Total word pairs in the graph = {}".format(len(x)))
        plt.savefig("../output/accuracy_confidence_plot.png", dpi=100, quality=100)
        plt.savefig("../output/accuracy_confidence_plot.png", dpi=100, quality=100)


def plot_accuracy_confidence_histogram(x, y):
    import numpy as np
    import matplotlib.pyplot as plt
    # makes the data
    conf_list = []
    acc_list = []

    for pair in y:
        conf_list.append(round(pair[1], 1))
        if pair[0] == 1:
            acc_list.append(round(pair[1], 1))

    if conf_list and acc_list:
        y1 = np.asarray(conf_list)
        y2 = np.asarray(acc_list)
        print(y1)
        print(y2)
        colors = ['b', 'g']
        # plots the histogram
        fig, ax1 = plt.subplots()
        ax1.hist([y1, y2], color=colors)
        ax1.set_xlim(0, 1)
        ax1.set_ylabel("Count")
        # plt.show()
        plt.savefig("../output/accuracy_confidence_plot_histogram_200.png", dpi=100, quality=100)
        plt.close()


def plot_accuracy_confidence_ratio_histogram(y):
    conf_list = []
    acc_list = []
    for pair in y:
        conf_list.append(round(pair[1], 1))

        if pair[0] == 1:
            acc_list.append(round(pair[1], 1))
    counter_acclist = Counter(acc_list)
    counter_conflist = Counter(conf_list)

    x_list = []
    y_list = []
    for k, v in counter_conflist.items():
        x_list.append(k)
        acc_count_for_k = counter_acclist.get(k, 0)
        ratio = acc_count_for_k / v

        y_list.append(ratio)
    width = 0.005
    fig, ax = plt.subplots()
    if len(conf_list) > 1:
        plt.bar(x_list,
                y_list,
                # of width
                width,
                # with alpha 0.5
                alpha=0.5,
                # with color
                color='b',
                # with label the second value in first_name
                label=x_list[0])

    # Set the y axis label
    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy Estimator')

    # Set the chart's title
    ax.set_title('Accuracy vs Confidence')
    plt.legend(['Confidence', 'Accuracy'], loc='upper right')
    plt.savefig("accuracy_confidence_plot_bar_{}.png".format(int(time.time())), dpi=100,
                quality=100)
    plt.close()


def plot_accuracy_percentage_conf_bin(y):
    print(y)
    conf_list = []
    acc_list = []
    for pair in y:
        conf_list.append(round(pair[1], 2))

        if pair[0] == 1:
            acc_list.append(round(pair[1], 2))
    counter_acclist = Counter(acc_list)
    counter_conflist = Counter(conf_list)

    x_list = []
    y_list = []
    x_ticks = []
    for k, v in counter_conflist.items():
        x_list.append(k)
        acc_count_for_k = counter_acclist.get(k, 0)
        percentage_accuracy_bin = acc_count_for_k / v
        y_list.append(percentage_accuracy_bin)
        x_ticks.append("{}/{}".format(acc_count_for_k, v))
    width = 0.005
    fig, ax = plt.subplots()
    if len(conf_list) > 1:
        plt.bar(x_list,
                y_list,
                # of width
                width,
                # with alpha 0.5
                alpha=0.5,
                # with color
                color='b',
                # with label the second value in first_name
                label=x_list[0])

    # Set the y axis label
    ax.set_xlabel('Confidence bins')
    ax.set_ylabel('Accuracy Percentage')

    # Set the chart's title
    ax.set_title('Accuracy vs Confidence')
    # ax.set_xticks(x_ticks
    totals = []
    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_height())

    # set individual bar lables using above list
    total = sum(totals)

    # set individual bar lables using above list
    index = 0
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x(), i.get_height(), x_ticks[index], fontsize=11,
                color='dimgrey')
        index += 1
    # ax.set_xticklabels(labels)
    # plt.legend(['counts_acc1/total_counts', 'Accuracy'], loc='upper left')
    plt.savefig("acccuracy1_percentage_bin_{}.png".format(int(time.time())), dpi=100,
                quality=100)
    plt.close()

def plot_accuracy_confidence_boxplot(y):
    conf_list = []
    bin1 = []
    bin2 = []
    bin3 = []
    bin4 = []
    bin5 = []
    bin6 = []
    bin7 = []
    bin8 = []
    bin9 = []
    bin1_countAcc = 0
    bin2_countAcc = 0
    bin3_countAcc = 0
    bin4_countAcc = 0
    bin5_countAcc = 0
    bin6_countAcc = 0
    bin7_countAcc = 0
    bin8_countAcc = 0
    bin9_countAcc = 0

    for pair in y:
        conf_list.append(round(pair[1], 3))
    print("y", y)
    for pair in y:
        if pair[1] >= 0.1 and pair[1] < 0.2:
            bin1.append(pair[1])
            if pair[0] == 1:
                bin1_countAcc += 1
        if pair[1] >= 0.2 and pair[1] < 0.3:
            bin2.append(pair[1])
            if pair[0] == 1:
                bin2_countAcc += 1
        if pair[1] >= 0.3 and pair[1] < 0.4:
            bin3.append(pair[1])
            if pair[0] == 1:
                bin3_countAcc += 1
        if pair[1] >= 0.4 and pair[1] < 0.5:
            bin4.append(pair[1])
            if pair[0] == 1:
                bin4_countAcc += 1
        if pair[1] >= 0.5 and pair[1] < 0.6:
            bin5.append(pair[1])
            if pair[0] == 1:
                bin5_countAcc += 1
        if pair[1] >= 0.6 and pair[1] < 0.7:
            bin6.append(pair[1])
            if pair[0] == 1:
                bin6_countAcc += 1
        if pair[1] >= 0.7 and pair[1] < 0.8:
            bin7.append(pair[1])
            if pair[0] == 1:
                bin7_countAcc += 1
        if pair[1] >= 0.8 and pair[1] < 0.9:
            bin8.append(pair[1])
            if pair[0] == 1:
                bin8_countAcc += 1
        if pair[1] >= 0.9 and pair[1] <= 1:
            bin9.append(pair[1])
            if pair[0] == 1:
                bin9_countAcc += 1

    size_bin1 = len(bin1)
    size_bin2 = len(bin2)
    size_bin3 = len(bin3)
    size_bin4 = len(bin4)
    size_bin5 = len(bin5)
    size_bin6 = len(bin6)
    size_bin7 = len(bin7)
    size_bin8 = len(bin8)
    size_bin9 = len(bin9)
    print("bin7", bin7)

    prob_success_dict = {}
    prob_success_bin1 = prob_success_bin2 = prob_success_bin3 = prob_success_bin4 = prob_success_bin5 = prob_success_bin6 = prob_success_bin7 = prob_success_bin8 = prob_success_bin9 = 0
    dist1 = dist2 = dist3 = dist4 = dist5 = dist6 = dist7 = dist8 = dist9 = []
    if size_bin1 != 0:
        prob_success_bin1 = bin1_countAcc / size_bin1
        prob_success_dict.update({0.1: prob_success_bin1})
    if size_bin2 != 0:
        prob_success_bin2 = bin2_countAcc / size_bin2
        prob_success_dict.update({0.2: prob_success_bin2})

    if size_bin3 != 0:
        prob_success_bin3 = bin3_countAcc / size_bin3
        prob_success_dict.update({0.3: prob_success_bin3})

    if size_bin4 != 0:
        prob_success_bin4 = bin4_countAcc / size_bin4
        prob_success_dict.update({0.4: prob_success_bin4})

    if size_bin5 != 0:
        prob_success_bin5 = bin5_countAcc / size_bin5
        prob_success_dict.update({0.5: prob_success_bin5})

    if size_bin6 != 0:
        prob_success_bin6 = bin6_countAcc / size_bin6
        prob_success_dict.update({0.6: prob_success_bin6})

    if size_bin7 != 0:
        prob_success_bin7 = bin7_countAcc / size_bin7
        prob_success_dict.update({0.7: prob_success_bin7})

    if size_bin8 != 0:
        prob_success_bin8 = bin8_countAcc / size_bin8
        prob_success_dict[0.8] = prob_success_bin8

    if size_bin9 != 0:
        prob_success_bin9 = bin9_countAcc / size_bin9
        prob_success_dict[0.9] = prob_success_bin9

    len(prob_success_dict)

    for k, v in prob_success_dict.items():
        if k == 0.1:
            dist1 = np.random.binomial(len(bin1), prob_success_bin1, size=bin1_countAcc)

        if k == 0.2:
            dist2 = np.random.binomial(len(bin2), prob_success_bin2, size=bin2_countAcc)
        if k == 0.3:
            dist3 = np.random.binomial(len(bin3), prob_success_bin3, size=bin3_countAcc)
        if k == 0.4:
            dist4 = np.random.binomial(len(bin4), prob_success_bin4, size=bin4_countAcc)
        if k == 0.5:
            dist5 = np.random.binomial(len(bin5), prob_success_bin5, size=bin5_countAcc)
        if k == 0.6:
            dist6 = np.random.binomial(len(bin6), prob_success_bin6, size=bin6_countAcc)
        if k == 0.7:
            dist7 = np.random.binomial(len(bin7), prob_success_bin7, size=bin7_countAcc)
            print(dist7)
        if k == 0.8:
            dist8 = np.random.binomial(len(bin8), prob_success_bin8, size=bin8_countAcc)
        if k == 0.9:
            dist9 = np.random.binomial(len(bin9), prob_success_bin9, size=bin9_countAcc)
    bin_size_list = [size_bin1,
                     size_bin2,
                     size_bin3,
                     size_bin4,
                     size_bin5,
                     size_bin6,
                     size_bin7,
                     size_bin8,
                     size_bin9, ]

    bin_count_acc_list = [bin1_countAcc,
                          bin2_countAcc,
                          bin3_countAcc,
                          bin4_countAcc,
                          bin5_countAcc,
                          bin6_countAcc,
                          bin7_countAcc,
                          bin8_countAcc,
                          bin9_countAcc]

    std_dev_for_bins = [0] * 10

    # for i in range(0, 9):
    #     if bin_size_list[i]:
    #         prob = bin_count_acc_list[i] / bin_size_list[i]
    #         std_dev_for_bins[i] = math.sqrt(prob * (1 - prob)) / math.sqrt(bin_size_list[i])

    # std_bins =[std_bin1,std_bin2,std_bin3,]
    data_to_plot = [dist1, dist2, dist3, dist4, dist5, dist6, dist7, dist8, dist9]
    data_to_plot2 = [bin1, bin2, bin3, bin4, bin5, bin6, bin7, bin8, bin9]
    # print('@@@@@@@@@@@@',data_to_plot)

    # Create a figure instance
    fig = plt.figure(1, figsize=(9, 6))

    # Create an axes instance
    ax = fig.add_subplot(111)

    # Create the boxplot
    ax.boxplot(data_to_plot)
    ax.set_title('Accuracy vs Confidence bins')
    ax.set_xlabel('Confidence bins')
    ax.set_ylabel('Box plot of Binomial distribution of Accuracy values ')

    # Save the figure
    fig.savefig('fig5.png', bbox_inches='tight')
    plt.close()


file_counter = 1
counter_2 = 1


def plot_each_accuracy_confidence_histogram(x, y):
    # import numpy as np
    # import matplotlib.pyplot as plttruelables

    # makes the data

    try:

        conf_list = []
        acc_list = []
        global file_counter
        for pair in y:
            conf_list.append(pair[1])
            acc_list.append(pair[0])
            # for pair in y:
            #     if pair[0] == 1:
            #         acc_list.append(pair[1])
            #     else:
            #         conf_list.append(pair[1])

            # if conf_list and acc_list:
            #
            #     conf_count = -10 if len(conf_list) > 9 else len(conf_list) * -1
            #     acc_count = -10 if len(acc_list) > 9 else len(acc_list) * -1

            y1 = conf_list
            # y1_size = len(conf_list)
            y2 = acc_list
            # y2_size = len(acc_list)
            print(y1)
            print(y2)
            colors = ['r', 'g']
            # plots the histogram
            fig, ax1 = plt.subplots()
            # if acc_count <0 :
            ax1.hist([y1, y2], color=colors)
            ax1.set_xlim(0, 1)
            ax1.set_ylabel("Count")
            # plt.show()
            global counter_2
            if file_counter == counter_2 * 10:
                plt.savefig("../output/accuracy_confidence_plot_histogram{}.png".format(counter_2), dpi=100,
                            quality=100)
                counter_2 = counter_2 + 1
            plt.close()
            file_counter = file_counter + 1
    except Exception as e:
        print(e)


def plot_histogram(y):
    # x = [
    #     datetime.datetime(2011, 1, 4, 0, 0),
    #     datetime.datetime(2011, 1, 5, 0, 0),
    #     datetime.datetime(2011, 1, 6, 0, 0)
    # ]
    # x = date2num(x)

    conf_list = []
    acc_list = []
    for pair in y:
        # if pair[0]==1:
        acc_list.append(pair[0])
        conf_list.append(round(pair[1], 2))
    pos = list(range(len(conf_list)))
    width = 0.25
    # ax = plt.subplot(111)
    len_conf_list = len(conf_list)
    # N=2
    # x = np.linspace(0,50,N)
    fig, ax = plt.subplots()
    z1 = conf_list
    y1 = acc_list
    # k = [11, 12, 13]
    # for i in range(0,len_conf_list):
    if len(conf_list) > 1:
        plt.bar(pos,
                # using df['pre_score'] data,
                z1,
                # of width
                width,
                # with alpha 0.5
                alpha=0.5,
                # with color
                color='r',
                # with label the first value in first_name
                label=conf_list[0])

        # Create a bar with mid_score data,
        # in position pos + some width buffer,
        plt.bar([p + width for p in pos],
                # using df['mid_score'] data,
                y1,
                # of width
                width,
                # with alpha 0.5
                alpha=0.5,
                # with color
                color='g',
                # with label the second value in first_name
                label=conf_list[1])

    # Set the y axis label
    ax.set_xlabel('Adjective-noun pair number')
    ax.set_ylabel('Confidence & Accuracy Value')

    # Set the chart's title
    ax.set_title('Accuracy and Confidence')

    # # Set the position of the x ticks
    # ax.set_xticks([p + 1.5 * width for p in pos])
    #
    # # Set the labels for the x ticks
    # ax.set_xticklabels(df['first_name'])

    # Setting the x-axis and y-axis limits
    if len(pos) != 0:
        plt.xlim(min(pos) - width, max(pos) + width * 4)
    # plt.ylim([0, max(df['pre_score'] + df['mid_score'] + df['post_score'])])

    # ax = plt.subplot(111)
    # ax.bar(x-0.2, y, width=0.2, color='b', align='center')
    # ax.bar(x, z, width=0.2, color='g', align='center')
    # ax.bar(x + 0.2, k, width=0.2, color='r', align='center')
    # ax.xaxis_date()
    # plt.xlim(min(pos) - width, max(pos) + width * 4)
    plt.legend(['Confidence', 'Accuracy'], loc='upper left')
    # plt.show()
    plt.savefig("../output/accuracy_confidence_plot_histogram_final_54_with0Accuracy_changedcolor.png", dpi=100,
                quality=100)
    plt.close()


truelabels = []
predictlabels = []


def calc_homogenity_comp_vmeas(user_input_df, candidates):
    # user_input_df['predict'] = y1

    confidence_counter = -1
    for c in candidates:
        confidence_counter += 1
        adj = c.getSource()
        noun = c.getTarget()
        candidate_df = user_input_df.loc[(user_input_df['adj'] == adj) & (user_input_df['noun'] == noun)]
        print(candidate_df["adj"][confidence_counter])
        print(candidate_df["noun"][confidence_counter])
        if candidate_df["class"][confidence_counter] != 2:
            truelabels.append(candidate_df["class"][confidence_counter])
            predictlabels.append(candidate_df["predict"][confidence_counter])
    print("truelables:", truelabels)
    print("predictlabels:", predictlabels)
    homogenity_scr = homogeneity_score(truelabels, predictlabels)
    vmeasure_scr = v_measure_score(truelabels, predictlabels)
    completness_scr = completeness_score(truelabels, predictlabels)
    print("homogenity_scr={},vmeasure_scr={},completness_scr={}".format(homogenity_scr, vmeasure_scr, completness_scr))


def calc_homogenity_comp_vmeas_training(df, y_train):
    # user_input_df['predict'] = y1

    # confidence_counter = -1
    # for c in candidates:
    #     confidence_counter += 1
    #     adj = c.getSource()
    #     noun = c.getTarget()
    #     candidate_df = user_input_df.loc[(user_input_df['adj'] == adj) & (user_input_df['noun'] == noun)]
    #     print(candidate_df["adj"][confidence_counter])
    #     print(candidate_df["noun"][confidence_counter])
    #     if candidate_df["class"][confidence_counter] != 2:
    #         truelabels.append(candidate_df["class"][confidence_counter])
    #         predictlabels.append(candidate_df["predict"][confidence_counter])
    # print("truelables:",truelabels)
    # print("predictlabels:",predictlabels)
    # homogenity_scr = homogeneity_score(truelabels,predictlabels)
    # vmeasure_scr = v_measure_score(truelabels,predictlabels)
    # completness_scr =completeness_score(truelabels,predictlabels)
    # print("homogenity_scr={},vmeasure_scr={},completness_scr={}".format(homogenity_scr,vmeasure_scr,completness_scr))

    truelabels = df['class']
    predictlabels = y_train
    homogenity_scr = homogeneity_score(truelabels, predictlabels)
    vmeasure_scr = v_measure_score(truelabels, predictlabels)
    completness_scr = completeness_score(truelabels, predictlabels)
    print("truelables:", truelabels)
    print("predictlabels:", predictlabels)
    print("homogenity_scr={},vmeasure_scr={},completness_scr={}".format(homogenity_scr, vmeasure_scr, completness_scr))
