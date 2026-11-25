import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

dataset_wine = load_wine()
df_x = pd.DataFrame(dataset_wine.data, columns=dataset_wine.feature_names)
s_y = pd.Series(dataset_wine.target, name="True_Class")

print("Wine dataset loaded!")
print("Shape:", df_x.shape)
print("\nSample rows:")
print(df_x.head())
print("\nClasses:", dataset_wine.target_names)

std_scaler = StandardScaler()
x_scaled = std_scaler.fit_transform(df_x)

pca_2d = PCA(n_components=2, random_state=42)
x_pca = pca_2d.fit_transform(x_scaled)

var_ratios = pca_2d.explained_variance_ratio_
retained_pct = sum(var_ratios) * 100

print("\nExplained variance ratio:", var_ratios)
print(f"Variance preserved in 2 components: {retained_pct:.2f}%")

viz_df = pd.DataFrame(x_pca, columns=["PC1", "PC2"])
viz_df["True_Class"] = s_y

km_cluster = KMeans(n_clusters=3, random_state=42, n_init=10)
pred_clusters = km_cluster.fit_predict(x_pca)
viz_df["Cluster"] = pred_clusters

sil_val = silhouette_score(x_pca, pred_clusters)
print("\nSilhouette Score:", round(sil_val, 4))

print("\nCluster vs True Class Crosstab:")
print(pd.crosstab(viz_df["Cluster"], viz_df["True_Class"]))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

sns.scatterplot(
    x="PC1",
    y="PC2",
    hue="Cluster",
    palette="coolwarm",
    data=viz_df,
    s=70,
    ax=ax1,
)
centers = km_cluster.cluster_centers_
ax1.scatter(
    centers[:, 0],
    centers[:, 1],
    c="black",
    marker="P",
    s=180,
    label="Cluster Centers",
)
ax1.set_title("2D PCA with K-Means Clustering")
ax1.legend(loc="best")

sns.scatterplot(
    x="PC1",
    y="PC2",
    hue="True_Class",
    palette="deep",
    data=viz_df,
    s=70,
    ax=ax2,
)
ax2.set_title("2D PCA with Ground Truth Classes")

plt.tight_layout()
plt.show()