import pandas as pd

df = pd.read_csv("results.csv")

total = len(df)

print("Total Questions:", total)

threshold0 = 0.60
hallucinations0 = df[df["Final Score"] > threshold0]

detected0 = len(hallucinations0)
rate0 = detected0 / total

print("\nThreshold:", threshold0)
print("Hallucinations Detected:", detected0)
print("Detection Rate:", round(rate0,3))

threshold1 = 0.65
hallucinations1 = df[df["Final Score"] > threshold1]

detected1 = len(hallucinations1)
rate1 = detected1 / total

print("\nThreshold:", threshold1)
print("Hallucinations Detected:", detected1)
print("Detection Rate:", round(rate1,3))

threshold2 = 0.70
hallucinations2 = df[df["Final Score"] > threshold2]

detected2 = len(hallucinations2)
rate2 = detected2 / total

print("\nThreshold:", threshold2)
print("Hallucinations Detected:", detected2)
print("Detection Rate:", round(rate2,3))

# Correction success rate (same as strict threshold)
corrections = df[df["Final Score"] > 0.65]

correction_rate = len(corrections) / total

print("\nCorrection Success Rate:", round(correction_rate,3))

from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score

df = pd.read_csv("results.csv")

y_true = df["True Label"]
y_pred = df["Detected"].astype(int)
y_score = df["Final Score"]

print("\n--- Additional Evaluation (SelfCheckGPT Style) ---")

print("Precision:", round(precision_score(y_true, y_pred),3))
print("Recall:", round(recall_score(y_true, y_pred),3))
print("F1 Score:", round(f1_score(y_true, y_pred),3))
print("AUC Score:", round(roc_auc_score(y_true, y_score),3))

from sklearn.metrics import precision_score, recall_score, f1_score

# SelfCheck predictions
selfcheck_detected = df["SelfCheck Score"] > 0.65

self_precision = precision_score(df["True Label"], selfcheck_detected)
self_recall = recall_score(df["True Label"], selfcheck_detected)
self_f1 = f1_score(df["True Label"], selfcheck_detected)
print("\n--- SelfCheckGPT Metrics ---")
print("Precision:", round(self_precision,3))
print("Recall:", round(self_recall,3))
print("F1 Score:", round(self_f1,3))

self_auc = roc_auc_score(df["True Label"], df["SelfCheck Score"])
print("AUC Score:", round(self_auc,3))