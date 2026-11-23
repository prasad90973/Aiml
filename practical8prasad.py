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

bc_data = load_breast_cancer()
df_x = pd.DataFrame(bc_data.data, columns=bc_data.feature_names)
s_y = pd.Series(bc_data.target)

print("Breast Cancer dataset loaded!")
print("Shape:", df_x.shape)
print("\nSample rows:")
print(df_x.head())
print("\nClasses:", bc_data.target_names)
print("0 = Malignant, 1 = Benign")

train_x, test_x, train_y, test_y = train_test_split(
    df_x,
    s_y,
    test_size=0.20,
    random_state=42,
    stratify=s_y,
)

print("\nTrain count:", train_x.shape[0])
print("Test count:", test_x.shape[0])

scaler_obj = StandardScaler()
train_x_std = scaler_obj.fit_transform(train_x)
test_x_std = scaler_obj.transform(test_x)

lr_model = LogisticRegression(max_iter=5000)
lr_model.fit(train_x_std, train_y)
print("\nModel training complete!")

y_hat = lr_model.predict(test_x_std)
y_scores = lr_model.predict_proba(test_x_std)[:, 1]

score_acc = accuracy_score(test_y, y_hat)
print("\nAccuracy:", round(score_acc * 100, 2), "%")

matrix_cm = confusion_matrix(test_y, y_hat)
print("\nConfusion Matrix:")
print(matrix_cm)

plt.figure(figsize=(6, 4))
sns.heatmap(
    matrix_cm,
    annot=True,
    fmt="d",
    cmap="Purples",
    xticklabels=bc_data.target_names,
    yticklabels=bc_data.target_names,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix Heatmap")
plt.tight_layout()
plt.show()

print("\nClassification Report:")
print(
    classification_report(
        test_y,
        y_hat,
        target_names=bc_data.target_names,
    )
)

false_pos_rate, true_pos_rate, _ = roc_curve(test_y, y_scores)
roc_auc = roc_auc_score(test_y, y_scores)

plt.figure(figsize=(7, 5))
plt.plot(false_pos_rate, true_pos_rate, label=f"ROC Curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], "k--", label="Baseline")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve - Tumor Classification")
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.show()

print("ROC-AUC:", round(roc_auc, 4))