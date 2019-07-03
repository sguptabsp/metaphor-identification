import pandas as pd

csv_path = "../data/TroFiMetaphors.csv"

metaphor_df = pd.read_csv(csv_path)

false_positive = metaphor_df[metaphor_df.type == "false positive"].shape[0]
false_negative = metaphor_df[metaphor_df.type == "false negative"].shape[0]
true_positive = metaphor_df[metaphor_df.type == "true positive"].shape[0]
true_negative = metaphor_df[metaphor_df.type == "true negative"].shape[0]
unknown = metaphor_df[metaphor_df.type == "unknown"].shape[0]
print("count_false_positive:", false_positive)
print("count_false_negative:", false_negative)
print("count_true_positive:", true_positive)
print("count_true_negative:", true_negative)
print("count_unknown", unknown)


precision = true_positive / (true_positive + false_positive)
recall = true_positive / (true_positive + false_negative)
print("precision:",precision)
print("recall:",recall)