import pandas as pd

results = pd.read_csv('./Sample/data/clustering/results.csv')

print(len(set(results['verb'].values.tolist())))
print(results.shape[0])

results = pd.read_csv('./Sample/data/clustering/results2.csv')

print(len(set(results['verb'].values.tolist())))
print(results.shape[0])