import pandas as pd

data_csv_path = "./data/TroFiMetaphorDataset.csv"
results_csv_path = "./data/TroFiMetaphorFullResults.csv"
write_csv_path = "./data/TroFiMetaphorUsefullResults.csv"


data = pd.read_csv(data_csv_path)
results = pd.read_csv(results_csv_path)

n_results = results.shape[0]

verbs = set(data["verb"].tolist())

# Eliminating the results with a source not in the dataset
offset = 0
for i in range(0, n_results):
    print(i)
    if results.iloc[i - offset]['source'] not in verbs:
        # print(results.iloc[i - offset]['source'])
        results = results.drop(results.index[i - offset])
        offset += 1


results.to_csv(path_or_buf=write_csv_path, index=False)

