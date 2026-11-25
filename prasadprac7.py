import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
)

bc_bundle = load_breast_cancer()
input_df = pd.DataFrame(bc_bundle.data, columns=bc_bundle.feature_names)
label_series = pd.Series(bc_bundle.target)

print("Breast Cancer dataset loaded!")
print("Shape:", input_df.shape)
print("\nSample rows:")
print(input_df.head())
print("\nClasses:", bc_bundle.target_names)
print("0 = Malignant, 1 = Benign")

train_inputs, test_inputs, train_labels, test_labels = train_test_split(
    input_df,
    label_series,
    test_size=0.20,
    random_state=42,
    stratify=label_series,
)

print("\nTrain count:", train_inputs.shape[0])
print("Test count:", test_inputs.shape[0])

std_scaler = StandardScaler()
train_inputs_scaled = std_scaler.fit_transform(train_inputs)
test_inputs_scaled = std_scaler.transform(test_inputs)

log_reg = LogisticRegression(max_iter=5000)
log_reg.fit(train_inputs_scaled, train_labels)
print("\nModel training complete!")

pred_classes = log_reg.predict(test_inputs_scaled)
pred_scores = log_reg.predict_proba(test_inputs_scaled)[:, 1]

model_acc = accuracy_score(test_labels, pred_classes)
print("\nAccuracy:", round(model_acc * 100, 2), "%")

conf_matrix = confusion_matrix(test_labels, pred_classes)
print("\nConfusion Matrix:")
print(conf_matrix)

plt.figure(figsize=(6, 4))
sns.heatmap(
    conf_matrix,
    annot=True,
    fmt="d",
    cmap="Greens",
    xticklabels=bc_bundle.target_names,
    yticklabels=bc_bundle.target_names,
)
plt.xlabel("Predicted Label")
plt.ylabel("Actual Label")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(
    classification_report(
        test_labels,
        pred_classes,
        target_names=bc_bundle.target_names,
    )
)

false_pos, true_pos, _ = roc_curve(test_labels, pred_scores)
roc_value = roc_auc_score(test_labels, pred_scores)

plt.figure(figsize=(7, 5))
plt.plot(false_pos, true_pos, label=f"ROC Curve (AUC = {roc_value:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Tumor Classification")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

print("ROC-AUC:", round(roc_value, 4))