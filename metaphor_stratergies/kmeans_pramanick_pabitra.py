import gensim
from gensim.models import Word2Vec

import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score
import numpy as np
import nltk

# Cosine Similarity model creation
from gensim.models import KeyedVectors
model = KeyedVectors.load_word2vec_format('wiki.en.vec')

# Cosign Train set
MET_AN_EN = pd.read_table('training_adj_noun_met_en.txt', delim_whitespace=True, names=('adj', 'noun'))
MET_AN_EN['class'] = 1
LIT_AN_EN = pd.read_table('training_adj_noun_nonmet_en.txt', delim_whitespace=True, names=('adj', 'noun'))
LIT_AN_EN['class'] = 0

df = pd.concat([LIT_AN_EN, MET_AN_EN])
df = pd.DataFrame(df)

an_vectorized = []

for j in zip(df.adj, df.noun):
    try:
        an_vectorized.append(model.similarity(j[0], j[1]))
    except:
        an_vectorized.append(0)

an_vectorized = pd.DataFrame(an_vectorized)
print(an_vectorized)
kmeans_clustering = KMeans(n_clusters=2, random_state=43)
idx = kmeans_clustering.fit_predict(an_vectorized)
print('Accuracy is: ', accuracy_score(idx, np.asarray(df['class'])))

# Cosign Test set

fields = ['adj', 'noun']
MET_AN_EN_TEST = pd.read_excel('Datasets_ACL2014.xlsx', sheetname='MET_AN_EN', usecols=fields)
MET_AN_EN_TEST['class'] = 1
LIT_AN_EN_TEST = pd.read_excel('Datasets_ACL2014.xlsx', sheetname='LIT_AN_EN', usecols=fields)
LIT_AN_EN_TEST['class'] = 0

MET_AN_EN_TRAIN = pd.read_table('training_adj_noun_met_en.txt', delim_whitespace=True, names=('adj', 'noun'))
MET_AN_EN_TRAIN['class'] = 1
LIT_AN_EN_TRAIN = pd.read_table('training_adj_noun_nonmet_en.txt', delim_whitespace=True, names=('adj', 'noun'))
LIT_AN_EN_TRAIN['class'] = 0

df = pd.concat([LIT_AN_EN_TEST, MET_AN_EN_TEST])
df = pd.DataFrame(df)

# df = pd.concat([LIT_AN_EN_TRAIN,MET_AN_EN_TRAIN])
# df = pd.DataFrame(df)

# creating word2vec model using dataset
data = []
for j in zip(df.adj, df.noun):
    temp = [j[0], j[1]]
    data.append(temp)
model = gensim.models.Word2Vec(data, min_count=1, size=200, window=5)

an_vectorized = []

for j in zip(df.adj, df.noun):
    an_vectorized.append(model.similarity(j[0], j[1]))

an_vectorized = pd.DataFrame(an_vectorized)
kmeans_clustering = KMeans(n_clusters=2, random_state=43)
idx = kmeans_clustering.fit_predict(an_vectorized)
print('Accuracy is: ', accuracy_score(idx, np.asarray(df['class'])))

#Edit Distance train set

MET_AN_EN = pd.read_table('training_adj_noun_met_en.txt', delim_whitespace=True, names=('adj', 'noun'))
MET_AN_EN['class'] = 1
LIT_AN_EN = pd.read_table('training_adj_noun_nonmet_en.txt', delim_whitespace=True, names=('adj', 'noun'))
LIT_AN_EN['class'] = 0

df = pd.concat([LIT_AN_EN,MET_AN_EN])
df = pd.DataFrame(df)
l=[]
an_vectorized = []
for j in zip(df.adj,df.noun):
    an_vectorized.append(nltk.edit_distance(j[0],j[1])/10)
    #an_vectorized.append(list(l))
# an_vectorized = np.asarray(an_vectorized)

an_vectorized = pd.DataFrame(an_vectorized)
kmeans_clustering = KMeans( n_clusters = 2, random_state=45)
idx = kmeans_clustering.fit_predict( an_vectorized )
print('Accuracy is: ',accuracy_score(idx,np.asarray(df['class'])))

# Edit Test Set

fields = ['adj', 'noun']
MET_AN_EN_TEST = pd.read_excel('Datasets_ACL2014.xlsx', sheetname='MET_AN_EN', usecols=fields)
MET_AN_EN_TEST['class'] = 1
LIT_AN_EN_TEST = pd.read_excel('Datasets_ACL2014.xlsx', sheetname='LIT_AN_EN', usecols=fields)
LIT_AN_EN_TEST['class'] = 0

df = pd.concat([LIT_AN_EN_TEST, MET_AN_EN_TEST])
df = pd.DataFrame(df)
# s1=[]
an_vectorized1 = []
an_vectorized = []
for j in zip(df.adj, df.noun):
    #     s1.append(nltk.edit_distance(j[0],j[1])/10)
    #     an_vectorized.append(list(s1))
    an_vectorized.append(nltk.edit_distance(j[0], j[1]) / 10)
    # an_vectorized.append(( ar_Adj + ar_Noun+ np.sign(ar_Adj - ar_Noun))/10)
# an_vectorized1_shape = np.reshape(an_vectorized1)
# an_vectorized = np.asarray(an_vectorized1_shape)

an_vectorized = pd.DataFrame(an_vectorized)
kmeans_clustering = KMeans(n_clusters=2, random_state=0)
idx = kmeans_clustering.fit_predict(an_vectorized)
print('Accuracy is: ', accuracy_score(idx, np.asarray(df['class'])))

# Load Abstractness Rating


csv = pd.read_csv("AC_ratings_google3m_koeper_SiW.csv",error_bad_lines=False)
csv = pd.DataFrame(csv)
dict = {}

for index,row in csv.iterrows():
    s = (row['WORD\tRATING']).split('\t')
    dict[s[0]] = s[1]
    #print(dict[s[0]])

#Abstractness Train set

MET_AN_EN = pd.read_table('training_adj_noun_met_en.txt', delim_whitespace=True, names=('adj', 'noun'))
MET_AN_EN['class'] = 1
LIT_AN_EN = pd.read_table('training_adj_noun_nonmet_en.txt', delim_whitespace=True, names=('adj', 'noun'))
LIT_AN_EN['class'] = 0

df = pd.concat([LIT_AN_EN,MET_AN_EN])
df = pd.DataFrame(df)

data=[]
for j in zip(df.adj,df.noun):
    temp = [j[0],j[1]]
    data.append(temp)
model = gensim.models.Word2Vec(data, min_count = 1,size = 200, window = 5)
#model = KeyedVectors.load_word2vec_format('wiki.en.vec')

an_vectorized = []
for j in zip(df.adj,df.noun):
    a = j[0]
    n = j[1]
    l = []
    if '-' in a:
        s = a.split('-')
        ar_Adj = (float(dict[s[0]]) + float(dict[s[1]]))/2
    else:
        ar_Adj = float(dict[a])
    if '-' in n:
        s = n.split('-')
        ar_Noun = (float(dict[s[0]]) + float(dict[s[1]]))/2
    else:
        ar_Noun = float(dict[n])
    l.append(( ar_Adj)/10)
    l.append(( ar_Noun)/10)
    l.append((np.sign(ar_Adj - ar_Noun)))
    l.append(model.similarity(j[0],j[1]))
    l.append(nltk.edit_distance(j[0],j[1])/10)

    #an_vectorized = np.asarray([ [(ar_Adj)/10],[(ar_Noun)/10], [np.sign(ar_Adj - ar_Noun)]])
    an_vectorized.append(list(l))
    #an_vectorized.append(( ar_Adj + ar_Noun+ np.sign(ar_Adj - ar_Noun))/10)
an_vectorized = np.asarray(an_vectorized)
#an_vectorized = pd.DataFrame(an_vectorized)
kmeans_clustering = KMeans( n_clusters = 2, random_state=43)
idx = kmeans_clustering.fit_predict( an_vectorized )
print(an_vectorized.shape)
print('Accuracy is: ',accuracy_score(idx,np.asarray(df['class'])))

#Abstractness Test Set

fields = ['adj', 'noun']
MET_AN_EN_TEST=pd.read_excel('Datasets_ACL2014.xlsx',sheetname='MET_AN_EN', usecols=fields)
MET_AN_EN_TEST['class'] = 1
LIT_AN_EN_TEST=pd.read_excel('Datasets_ACL2014.xlsx',sheetname='LIT_AN_EN', usecols=fields)
LIT_AN_EN_TEST['class'] = 0

df = pd.concat([LIT_AN_EN_TEST,MET_AN_EN_TEST])
df = pd.DataFrame(df)
data=[]
for j in zip(df.adj,df.noun):
    temp = [j[0],j[1]]
    data.append(temp)

model = gensim.models.Word2Vec(data, min_count = 1,size = 200, window = 5)

an_vectorized = []
for j in zip(df.adj,df.noun):
    a = j[0]
    n = j[1]
    l = []
    if '-' in a:
        s = a.split('-')
        ar_Adj = (float(dict[s[0]]) + float(dict[s[1]]))/2
    else:
        ar_Adj = float(dict[a])
    if '-' in n:
        s = n.split('-')
        ar_Noun = (float(dict[s[0]]) + float(dict[s[1]]))/2
    else:
        ar_Noun = float(dict[n])
    l.append(( ar_Adj)/10)
    l.append(( ar_Noun)/10)
    l.append((np.sign(ar_Adj - ar_Noun)))
    l.append(model.similarity(j[0],j[1]))
    l.append(nltk.edit_distance(j[0],j[1])/10)

    #an_vectorized = np.asarray([ [(ar_Adj)/10],[(ar_Noun)/10], [np.sign(ar_Adj - ar_Noun)]])
    an_vectorized.append(list(l))
    #an_vectorized.append(( ar_Adj + ar_Noun)/10)
an_vectorized = np.asarray(an_vectorized)
#an_vectorized = pd.DataFrame(an_vectorized)
kmeans_clustering = KMeans( n_clusters = 2, random_state=43 )
idx = kmeans_clustering.fit_predict( an_vectorized )
print('Accuracy is: ',accuracy_score(idx,np.asarray(df['class'])))

## Merged Data

fields = ['adj', 'noun']
MET_AN_EN_TEST=pd.read_excel('Datasets_ACL2014.xlsx',sheetname='MET_AN_EN', usecols=fields)
MET_AN_EN_TEST['class'] = 1
LIT_AN_EN_TEST=pd.read_excel('Datasets_ACL2014.xlsx',sheetname='LIT_AN_EN', usecols=fields)
LIT_AN_EN_TEST['class'] = 0
MET_AN_EN = pd.read_table('training_adj_noun_met_en.txt', delim_whitespace=True, names=('adj', 'noun'))
MET_AN_EN['class'] = 1
LIT_AN_EN = pd.read_table('training_adj_noun_nonmet_en.txt', delim_whitespace=True, names=('adj', 'noun'))
LIT_AN_EN['class'] = 0

df = pd.concat([LIT_AN_EN_TEST,MET_AN_EN_TEST,LIT_AN_EN,MET_AN_EN])
df = pd.DataFrame(df)

an_vectorized = []
for j in zip(df.adj,df.noun):
    a = j[0]
    n = j[1]
    l = []
    if '-' in a:
        s = a.split('-')
        ar_Adj = (float(dict[s[0]]) + float(dict[s[1]]))/2
    else:
        ar_Adj = float(dict[a])
    if '-' in n:
        s = n.split('-')
        ar_Noun = (float(dict[s[0]]) + float(dict[s[1]]))/2
    else:
        ar_Noun = float(dict[n])
    #an_vectorized.append(( ar_Adj + ar_Noun)/10)
    l.append(( ar_Adj)/10)
    l.append(( ar_Noun)/10)
    l.append((np.sign(ar_Adj - ar_Noun)))
    l.append(nltk.edit_distance(j[0],j[1])/10)

    #an_vectorized = np.asarray([ [(ar_Adj)/10],[(ar_Noun)/10], [np.sign(ar_Adj - ar_Noun)]])
    an_vectorized.append(list(l))
    #an_vectorized.append(( ar_Adj + ar_Noun)/10)
an_vectorized = np.asarray(an_vectorized)
#an_vectorized = pd.DataFrame(an_vectorized)
kmeans_clustering = KMeans( n_clusters = 2, random_state=43 )
idx = kmeans_clustering.fit_predict( an_vectorized )
print('Accuracy is: ',accuracy_score(idx,np.asarray(df['class'])))