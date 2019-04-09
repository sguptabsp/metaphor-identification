import gensim
import nltk
import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

from new_structure.modules.datastructs.metaphor_group import MetaphorGroup


def create_word2vec_model():
    # Cosine Similarity model creation
    model = KeyedVectors.load_word2vec_format(
        '/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/wiki.en.vec')

    return model


def identify_metaphors_abstractness_cosine_edit_dist(candidates, cand_type, verbose):
    results = MetaphorGroup()

    MET_AN_EN = pd.read_table(
        '/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/training_adj_noun_met_en.txt',
        delim_whitespace=True, names=('adj', 'noun'))
    MET_AN_EN['class'] = 1
    LIT_AN_EN = pd.read_table(
        '/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/training_adj_noun_nonmet_en.txt',
        delim_whitespace=True, names=('adj', 'noun'))
    LIT_AN_EN['class'] = 0

    df = pd.concat([LIT_AN_EN, MET_AN_EN])

    data_list = []

    for c in candidates:
        dataframe_data = {}
        dataframe_data["adj"] = c.getSource()
        dataframe_data["noun"] = c.getTarget()
        data_list.append(dataframe_data)
    df_test_data = pd.DataFrame.from_records(data_list)
    df = pd.concat([df_test_data, df], axis=0).reset_index()
    an_vectorized = []
    # Load Abstractness Rating
    model = KeyedVectors.load_word2vec_format(
        '/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/wiki.en.vec')
    data = []
    for j in zip(df.adj, df.noun):
        temp = [j[0], j[1]]
        data.append(temp)
    model = gensim.models.Word2Vec(data, min_count=1, size=200, window=5)

    csv = pd.read_csv(
        "/home/shrutitejus/iit/research_project/Research_Project/new_structure/modules/datastructs/AC_ratings_google3m_koeper_SiW.csv",
        error_bad_lines=False)
    csv = pd.DataFrame(csv)
    dict = {}

    for index, row in csv.iterrows():
        s = (row['WORD\tRATING']).split('\t')
        dict[s[0]] = s[1]
        # print(dict[s[0]])

    for j in zip(df.adj, df.noun):
        a = j[0]
        n = j[1]
        l = []
        if '-' in a:
            s = a.split('-')
            ar_Adj = (float(dict[s[0]]) + float(dict[s[1]])) / 2
        else:
            ar_Adj = float(dict[a])
        if '-' in n:
            s = n.split('-')
            ar_Noun = (float(dict[s[0]]) + float(dict[s[1]])) / 2
        else:
            ar_Noun = float(dict[n])
        # an_vectorized.append(( ar_Adj + ar_Noun)/10)
        l.append((ar_Adj) / 10)
        l.append((ar_Noun) / 10)
        l.append((np.sign(ar_Adj - ar_Noun)))
        s = model.similarity(j[0], j[1])
        l.append(s)
        l.append(nltk.edit_distance(j[0], j[1]) / 10)

        # an_vectorized = np.asarray([ [(ar_Adj)/10],[(ar_Noun)/10], [np.sign(ar_Adj - ar_Noun)]])
        an_vectorized.append(list(l))
        # an_vectorized.append(create_word2vec_model().similarity(j[0], j[1]))

        # an_vectorized.append(( ar_Adj + ar_Noun)/10)
    an_vectorized = np.asarray(an_vectorized)
    # an_vectorized = pd.DataFrame(an_vectorized)
    kmeans_clustering = KMeans(n_clusters=2, random_state=43)
    idx = kmeans_clustering.fit_predict(an_vectorized)
    print('Accuracy is: ', accuracy_score(idx, np.asarray(df['class'])))
