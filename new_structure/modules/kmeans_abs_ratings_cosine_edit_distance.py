import gensim
import nltk
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans

from new_structure.modules.datastructs.metaphor import Metaphor
from new_structure.modules.datastructs.metaphor_group import MetaphorGroup
from new_structure.utils import timeit


def create_word2vec_model():
    # Cosine Similarity model creation
    model = KeyedVectors.load_word2vec_format(
        '/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/wiki.en.vec')

    return model


def get_abstractness_rating():
    csv = pd.read_csv(
        "/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/AC_ratings_google3m_koeper_SiW.csv",
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


@timeit
def identify_metaphors_abstractness_cosine_edit_dist(candidates, cand_type, verbose: str) -> MetaphorGroup:
    results = MetaphorGroup()

    MET_AN_EN = pd.read_table(
        '/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/training_adj_noun_met_en.txt',
        delim_whitespace=True, names=('adj', 'noun'))
    MET_AN_EN['class'] = 1
    LIT_AN_EN = pd.read_table(
        '/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/training_adj_noun_nonmet_en.txt',
        delim_whitespace=True, names=('adj', 'noun'))
    LIT_AN_EN['class'] = 0

    fields = ['adj', 'noun']
    MET_AN_EN_TEST = pd.read_excel(
        '/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/Datasets_ACL2014.xlsx',
        sheetname='MET_AN_EN', usecols=fields)
    MET_AN_EN_TEST['class'] = 1
    LIT_AN_EN_TEST = pd.read_excel(
        '/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/Datasets_ACL2014.xlsx',
        sheetname='LIT_AN_EN', usecols=fields)
    LIT_AN_EN_TEST['class'] = 0

    df = pd.concat([LIT_AN_EN, MET_AN_EN, MET_AN_EN_TEST, LIT_AN_EN_TEST])

    data_list = []

    for c in candidates:
        dataframe_data = {}
        dataframe_data["adj"] = c.getSource()
        dataframe_data["noun"] = c.getTarget()
        data_list.append(dataframe_data)
    df_test_data = pd.DataFrame.from_records(data_list)
    df = pd.concat([df], axis=0).reset_index()
    user_input_df = pd.concat([df_test_data], axis=0).reset_index()

    an_vectorized = vectorize_data(df)
    an_vectorized_user_input = vectorize_data(user_input_df)

    kmeans_clustering = KMeans(n_clusters=2, random_state=43)
    idx = kmeans_clustering.fit(an_vectorized)
    y1 = idx.predict(an_vectorized_user_input)

    user_input_df['predict'] = y1
    for c in candidates:
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
            results.addMetaphor(Metaphor(c, result, 1.0))
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
