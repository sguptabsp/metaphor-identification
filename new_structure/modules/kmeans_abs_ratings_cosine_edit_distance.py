import gensim
import nltk
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
from scipy.spatial import distance
from new_structure.modules.datastructs.metaphor import Metaphor
from new_structure.modules.datastructs.metaphor_group import MetaphorGroup
from new_structure.utils import timeit


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
        # s = model.similarity(j[0], j[1])
        # l.append(s)
        # l.append(nltk.edit_distance(j[0], j[1]) / 10)

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
        # l.append(nltk.edit_distance(j[0], j[1]) / 10)

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


def get_confidence(an_vectorized,kmeans_clustering,test_data_coordinates,predicted_data_labels):
    confidence_dict={}
    # pca = PCA(n_components=2).fit(an_vectorized)
    # an_vectorized_PCA = PCA(n_components=2).fit_transform(an_vectorized)
    # an_vectorized_PCA = kmeans_clustering.transform(an_vectorized)
    centroids = kmeans_clustering.cluster_centers_
    clustering_labels = kmeans_clustering.labels_
    # centroids_transformed = kmeans_clustering.transform(centroids)
    #idx = kmeans_clustering.fit(an_vectorized)
    # clusters = kmeans_clustering.fit_predict(an_vectorized_PCA)
    # clusters = y1
    X_dist = kmeans_clustering.transform(an_vectorized) **2
    # an_vectorized_PCA_square = an_vectorized_PCA**2
    # do something useful...
    import pandas as pd
    # df_conf = pd.DataFrame(an_vectorized_PCA_square.sum(axis=1).round(2), columns=['sqdist'])
    # df_conf['label'] = clustering_labels

    # df_conf.head()
    # print(df_conf.tail(10))
    centroid_list=centroids.tolist()
    distances = []
    #for i in range(len(centroid_list[0])):
    for i, (cx, cy) in enumerate(centroids):
        mean_distance = k_mean_distance(an_vectorized, cx, cy, i, clustering_labels)
        #mean_distance = k_mean_distance(an_vectorized_PCA, centroid_list[0][i], centroid_list[1][i], i, clusters)
        distances.append(mean_distance)

    print(distances)
    #
    max_indices = []
    for label in np.unique(kmeans_clustering.labels_):
        X_label_indices = np.where(clustering_labels == label)[0]
        max_label_idx = X_label_indices[np.argmax(X_dist[clustering_labels == label].sum(axis=1))]
        max_indices.append(max_label_idx)

    print(len(max_indices))
    # an_vectorized_PCA[max_indices, 0], an_vectorized_PCA[max_indices, 1]
    cluster_0_farthest_point=an_vectorized[max_indices, 0]
    cluster_1_farthest_point=an_vectorized[max_indices, 1]
    cluster_0_max_distance=distance.euclidean(centroid_list[0], cluster_0_farthest_point.tolist())
    cluster_1_max_distance=distance.euclidean(centroid_list[1], cluster_1_farthest_point.tolist())
    test_data_coordinate_list=test_data_coordinates.tolist()
    predicted_label_list=predicted_data_labels.tolist()
    for i in range(len(test_data_coordinate_list)):
        if predicted_label_list[i] == 0 :
            data_point_center_distance=distance.euclidean(centroid_list[0], test_data_coordinate_list[i])
            confidence_dict[i]=(cluster_0_max_distance-data_point_center_distance)/cluster_0_max_distance
        elif predicted_label_list[i] == 1:
            data_point_center_distance=distance.euclidean(centroid_list[1], test_data_coordinate_list[i])
            confidence_dict[i]=(cluster_1_max_distance-data_point_center_distance)/cluster_1_max_distance

    print(an_vectorized[max_indices, 0], an_vectorized[max_indices, 1])
    # X_dist_farthestPoint1 = kmeans_clustering.transform(an_vectorized_PCA[max_indices]) ** 2
    # X_dist_farthestPoint2 = kmeans_clustering.transform(an_vectorized_PCA[max_indices]) ** 2
    # an_vectorized_PCA_square1 = X_dist_farthestPoint1 ** 2
    # an_vectorized_PCA_square2 = X_dist_farthestPoint2 ** 2
    #
    # print(an_vectorized_PCA_square1,an_vectorized_PCA_square2)
    # do something useful...
    return confidence_dict

@timeit
def identify_metaphors_abstractness_cosine_edit_dist(candidates, cand_type, verbose: str) -> MetaphorGroup:
    results = MetaphorGroup()

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
    # df = pd.concat([df], axis=0).reset_index()
    user_input_df = pd.concat([df_test_data], axis=0).reset_index()

    # an_vectorized = vectorize_data_abstractness(df)
    # an_vectorized_user_input = vectorize_data_abstractness(user_input_df)
    # an_vectorized = vectorize_data_abstractness_cosine(df)
    # an_vectorized_user_input = vectorize_data_abstractness_cosine(user_input_df)
    an_vectorized = vectorize_data(df)
    an_vectorized_user_input = vectorize_data(user_input_df)

    an_vectorized_training_PCA = PCA(n_components=2).fit_transform(an_vectorized)
    an_vectorized_test_PCA = PCA(n_components=2).fit_transform(an_vectorized_user_input)

    kmeans_clustering = KMeans(n_clusters=2, random_state=43)
    idx = kmeans_clustering.fit(an_vectorized_training_PCA)
    kmeans_cluster_centers = kmeans_clustering.cluster_centers_
    y1 = idx.predict(an_vectorized_test_PCA)

    label=kmeans_clustering.labels_

    print(label)

    #
    # an_vectorized_conf_df= pd.DataFrame
    # an_vectorized_conf_df = pd.concat([an_vectorized,an_vectorized_user_input])
    # an_vectorized_conf_df=pd.append(an_vectorized)
    # conf_df= df.append(user_input_df)
    # an_vectorized_conf = vectorize_data(conf_df)
    confidence =get_confidence(an_vectorized_training_PCA,kmeans_clustering,an_vectorized_test_PCA,y1)
    print("Confidence of the corresponding words are : {} ".format(confidence))
    import matplotlib.pyplot as plt
    plt.scatter(an_vectorized[:, 0], an_vectorized[:, 1], c=kmeans_clustering.labels_, cmap='rainbow')
    plt.show()
    print('Accuracy is: ', accuracy_score(np.asarray(user_input_df['class']), y1))
    user_input_df['predict'] = y1
    confidence_counter=-1
    for c in candidates:
        confidence_counter+=1
        adj = c.getSource()
        noun = c.getTarget()
        candidate_df = user_input_df.loc[(user_input_df['adj'] == adj) & (user_input_df['noun'] == noun)]
        if len(candidate_df.index):
            result_class = candidate_df.iloc[0]['predict']
            # for j in zip(df.adj,df.noun):
            #     adj_df = j[0]
            #     noun_df =j[1]
            # #resultrow=df.query('adj == {} & noun == {}'.format(adj,noun))
            #     if adj_df==adj and noun_df==noun:
            #         result_class = df['predict']
            if result_class.any() == 0:
                result = False
            else:
                result = True

            print(result)
            # float_conf =float(result_class)
            results.addMetaphor(Metaphor(c, result, confidence[confidence_counter]))
    return results






#
#
# @timeit
# def identify_metaphors_abstractness_cosine_training(candidates, cand_type, verbose: str) -> MetaphorGroup:
#     results = MetaphorGroup()
#
#     MET_AN_EN = pd.read_table(
#         '/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/training_adj_noun_met_en.txt',
#         delim_whitespace=True, names=('adj', 'noun'))
#     MET_AN_EN['class'] = 1
#     LIT_AN_EN = pd.read_table(
#         '/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/training_adj_noun_nonmet_en.txt',
#         delim_whitespace=True, names=('adj', 'noun'))
#     LIT_AN_EN['class'] = 0
#
#     df = pd.concat([LIT_AN_EN, MET_AN_EN])
#
#     data_list = []
#
#     for c in candidates:
#         dataframe_data = {}
#         dataframe_data["adj"] = c.getSource()
#         dataframe_data["noun"] = c.getTarget()
#         data_list.append(dataframe_data)
#     df_test_data = pd.DataFrame.from_records(data_list)
#     df = pd.concat([df_test_data, df], axis=0).reset_index()
#     an_vectorized = []
#     # Load Abstractness Rating
#     # model = KeyedVectors.load_word2vec_format(
#     #     '/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/wiki.en.vec')
#     data = []
#     for j in zip(df.adj, df.noun):
#         temp = [j[0], j[1]]
#         data.append(temp)
#     model = gensim.models.Word2Vec(data, min_count=1, size=200, window=5)
#
#     csv = pd.read_csv(
#         "/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/AC_ratings_google3m_koeper_SiW.csv",
#         error_bad_lines=False)
#     csv = pd.DataFrame(csv)
#     dict = {}
#     csv_df = pd.DataFrame(csv['WORD\tRATING'].str.split('\t', 1).tolist(),
#                           columns=['word', 'rating'])
#     dict=pd.Series(csv_df.rating.values, index=csv_df.word).to_dict()
#
#     # for index, row in csv.iterrows():
#     #     s = (row['WORD\tRATING']).split('\t')
#     #     dict[s[0]] = s[1]
#     #     if index % 10000 == 0:
#     #         print(dict[s[0]])
#
#     for j in zip(df.adj, df.noun):
#         a = j[0]
#         n = j[1]
#         l = []
#         if '-' in a:
#             s = a.split('-')
#             ar_Adj = (float(dict[s[0]]) + float(dict[s[1]])) / 2
#         else:
#             ar_Adj = float(dict[a])
#         if '-' in n:
#             s = n.split('-')
#             ar_Noun = (float(dict[s[0]]) + float(dict[s[1]])) / 2
#         else:
#             ar_Noun = float(dict[n])
#         # an_vectorized.append(( ar_Adj + ar_Noun)/10)
#         l.append((ar_Adj) / 10)
#         l.append((ar_Noun) / 10)
#         l.append((np.sign(ar_Adj - ar_Noun)))
#         s = model.similarity(j[0], j[1])
#         l.append(s)
#         # l.append(nltk.edit_distance(j[0], j[1]) / 10)
#
#         # an_vectorized = np.asarray([ [(ar_Adj)/10],[(ar_Noun)/10], [np.sign(ar_Adj - ar_Noun)]])
#         an_vectorized.append(list(l))
#         # an_vectorized.append(create_word2vec_model().similarity(j[0], j[1]))
#
#         # an_vectorized.append(( ar_Adj + ar_Noun)/10)
#     an_vectorized = np.asarray(an_vectorized)
#     # an_vectorized = pd.DataFrame(an_vectorized)
#     kmeans_clustering = KMeans(n_clusters=2, random_state=43)
#     idx = kmeans_clustering.fit_predict(an_vectorized)
#     #    print('Accuracy is: ', accuracy_score(idx, np.asarray(df['class'])))
#     df['predict'] = idx
#     for c in candidates:
#         adj = c.getSource()
#         noun = c.getTarget()
#         candidate_df = df.loc[(df['adj'] == adj) & (df['noun'] == noun)]
#         if len(candidate_df.index):
#             result_class = candidate_df.iloc[0]['predict']
#             # for j in zip(df.adj,df.noun):
#             #     adj_df = j[0]
#             #     noun_df =j[1]
#             # #resultrow=df.query('adj == {} & noun == {}'.format(adj,noun))
#             #     if adj_df==adj and noun_df==noun:
#             #         result_class = df['predict']
#             if result_class.any() == 0:
#                 result = False
#             else:
#                 result = True
#
#             print(result)
#             # float_conf =float(result_class)
#             results.addMetaphor(Metaphor(c, result, 1.0))
#     return results

# @timeit
# def identify_metaphors_abstractness_cosine_testing(candidates, cand_type, verbose: str) -> MetaphorGroup:
#     results = MetaphorGroup()
#
#     MET_AN_EN = pd.read_table(
#         '/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/training_adj_noun_met_en.txt',
#         delim_whitespace=True, names=('adj', 'noun'))
#     MET_AN_EN['class'] = 1
#     LIT_AN_EN = pd.read_table(
#         '/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/training_adj_noun_nonmet_en.txt',
#         delim_whitespace=True, names=('adj', 'noun'))
#     LIT_AN_EN['class'] = 0
#
#     df = pd.concat([LIT_AN_EN, MET_AN_EN])
#
#     data_list = []
#
#     for c in candidates:
#         dataframe_data = {}
#         dataframe_data["adj"] = c.getSource()
#         dataframe_data["noun"] = c.getTarget()
#         data_list.append(dataframe_data)
#     df_test_data = pd.DataFrame.from_records(data_list)
#     df = pd.concat([df_test_data, df], axis=0).reset_index()
#     an_vectorized = []
#     # Load Abstractness Rating
#     # model = KeyedVectors.load_word2vec_format(
#     #     '/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/wiki.en.vec')
#     data = []
#     for j in zip(df.adj, df.noun):
#         temp = [j[0], j[1]]
#         data.append(temp)
#     model = gensim.models.Word2Vec(data, min_count=1, size=200, window=5)
#
#     csv = pd.read_csv(
#         "/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/AC_ratings_google3m_koeper_SiW.csv",
#         error_bad_lines=False)
#     csv = pd.DataFrame(csv)
#     dict = {}
#     csv_df = pd.DataFrame(csv['WORD\tRATING'].str.split('\t', 1).tolist(),
#                           columns=['word', 'rating'])
#     dict=pd.Series(csv_df.rating.values, index=csv_df.word).to_dict()
#
#     # for index, row in csv.iterrows():
#     #     s = (row['WORD\tRATING']).split('\t')
#     #     dict[s[0]] = s[1]
#     #     if index % 10000 == 0:
#     #         print(dict[s[0]])
#
#     for j in zip(df.adj, df.noun):
#         a = j[0]
#         n = j[1]
#         l = []
#         if '-' in a:
#             s = a.split('-')
#             ar_Adj = (float(dict[s[0]]) + float(dict[s[1]])) / 2
#         else:
#             ar_Adj = float(dict[a])
#         if '-' in n:
#             s = n.split('-')
#             ar_Noun = (float(dict[s[0]]) + float(dict[s[1]])) / 2
#         else:
#             ar_Noun = float(dict[n])
#         # an_vectorized.append(( ar_Adj + ar_Noun)/10)
#         l.append((ar_Adj) / 10)
#         l.append((ar_Noun) / 10)
#         l.append((np.sign(ar_Adj - ar_Noun)))
#         s = model.similarity(j[0], j[1])
#         l.append(s)
#         # l.append(nltk.edit_distance(j[0], j[1]) / 10)
#
#         # an_vectorized = np.asarray([ [(ar_Adj)/10],[(ar_Noun)/10], [np.sign(ar_Adj - ar_Noun)]])
#         an_vectorized.append(list(l))
#         # an_vectorized.append(create_word2vec_model().similarity(j[0], j[1]))
#
#         # an_vectorized.append(( ar_Adj + ar_Noun)/10)
#     an_vectorized = np.asarray(an_vectorized)
#     # an_vectorized = pd.DataFrame(an_vectorized)
#     # kmeans_clustering = KMeans(n_clusters=2, random_state=43)
#     # idx = kmeans_clustering.fit_predict(an_vectorized)
#     #    print('Accuracy is: ', accuracy_score(idx, np.asarray(df['class'])))
#     kmeans_clustering = KMeans(n_clusters=2, random_state=43)
#     idx = kmeans_clustering.fit(an_vectorized)
#
#     # df['predict'] = idx
#
#     fields = ['adj', 'noun']
#     MET_AN_EN_TEST = pd.read_excel('/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/Datasets_ACL2014.xlsx', sheetname='MET_AN_EN', usecols=fields)
#     MET_AN_EN_TEST['class'] = 1
#     LIT_AN_EN_TEST = pd.read_excel('/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/Datasets_ACL2014.xlsx', sheetname='LIT_AN_EN', usecols=fields)
#     LIT_AN_EN_TEST['class'] = 0
#
#     df = pd.concat([LIT_AN_EN_TEST, MET_AN_EN_TEST])
#     df = pd.DataFrame(df)
#     data = []
#     for j in zip(df.adj, df.noun):
#         temp = [j[0], j[1]]
#         data.append(temp)
#
#     model = gensim.models.Word2Vec(data, min_count=1, size=200, window=5)
#
#     an_vectorized = []
#     for j in zip(df.adj, df.noun):
#         a = j[0]
#         n = j[1]
#         l = []
#         if '-' in a:
#             s = a.split('-')
#             ar_Adj = (float(dict[s[0]]) + float(dict[s[1]])) / 2
#         else:
#             ar_Adj = float(dict[a])
#         if '-' in n:
#             s = n.split('-')
#             ar_Noun = (float(dict[s[0]]) + float(dict[s[1]])) / 2
#         else:
#             ar_Noun = float(dict[n])
#         l.append((ar_Adj) / 10)
#         l.append((ar_Noun) / 10)
#         l.append((np.sign(ar_Adj - ar_Noun)))
#         l.append(model.similarity(j[0], j[1]))
#         l.append(nltk.edit_distance(j[0], j[1]) / 10)
#
#         # an_vectorized = np.asarray([ [(ar_Adj)/10],[(ar_Noun)/10], [np.sign(ar_Adj - ar_Noun)]])
#         an_vectorized.append(list(l))
#         # an_vectorized.append(( ar_Adj + ar_Noun)/10)
#     an_vectorized = np.asarray(an_vectorized)
#
#     # Y = idx.predict(an_vectorized)
#     # print('Accuracy is: ', accuracy_score(np.asarray(df['class']), Y))
#     # df['predict'] = Y
#
#     for c in candidates:
#         adj = c.getSource()
#         noun = c.getTarget()
#         candidate_df = df.loc[(df['adj'] == adj) & (df['noun'] == noun)]
#         if len(candidate_df.index):
#             result_class = candidate_df.iloc[0]['predict']
#             # for j in zip(df.adj,df.noun):
#             #     adj_df = j[0]
#             #     noun_df =j[1]
#             # #resultrow=df.query('adj == {} & noun == {}'.format(adj,noun))
#             #     if adj_df==adj and noun_df==noun:
#             #         result_class = df['predict']
#             if result_class.any() == 0:
#                 result = False
#             else:
#                 result = True
#
#             print(result)
#             # float_conf =float(result_class)
#             results.addMetaphor(Metaphor(c, result, 1.0))
#     return results
#
