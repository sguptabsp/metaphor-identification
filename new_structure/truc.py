import pandas as pd

# path = "./data/TroFiDataset.txt"
# data = pd.read_csv(path, sep='\t', names=['id', 'type', 'text'])
# print(data.iloc[49:52].values.tolist())


path_vn = "./data/vn2_v1.txt"
data = pd.read_csv(path_vn, sep='\t', names=['freq', 'v', 'n', 'pos1', 'pos2'])
print(data.columns)
print(data.shape)
useful_data = data[data['freq'] > 1000]
print(useful_data.columns)
print(useful_data.shape)
# useful_data.to_csv("./data/vn2.txt", header=None, index=None, sep='\t')
