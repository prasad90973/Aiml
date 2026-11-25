import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

raw_wine = load_wine()
df_features = pd.DataFrame(raw_wine.data, columns=raw_wine.feature_names)
s_classes = pd.Series(raw_wine.target, name="Class")

print("Wine dataset loaded!")
print("Shape:", df_features.shape)
print("\nSample rows:")
print(df_features.head())

std_scaler = StandardScaler()
x_norm = std_scaler.fit_transform(df_features)

num_clusters = 3
km_model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
assigned_clusters = km_model.fit_predict(x_norm)

score_sil = silhouette_score(x_norm, assigned_clusters)
print(f"\nSilhouette Score (k={num_clusters}):", round(score_sil, 4))

pca_2d = PCA(n_components=2, random_state=42)
coords_2d = pca_2d.fit_transform(x_norm)
centers_2d = pca_2d.transform(km_model.cluster_centers_)

plot_df = pd.DataFrame(coords_2d, columns=["PC1", "PC2"])
plot_df["Assigned_Cluster"] = assigned_clusters
plot_df["Class"] = s_classes

print("\nCrosstab of Clusters vs True Classes:")
print(pd.crosstab(plot_df["Assigned_Cluster"], plot_df["Class"]))

plt.figure(figsize=(8, 6))
sns.scatterplot(
    x="PC1",
    y="PC2",
    hue="Assigned_Cluster",
    palette="coolwarm",
    data=plot_df,
    s=75,
)
plt.scatter(
    centers_2d[:, 0],
    centers_2d[:, 1],
    c="black",
    marker="P",
    s=200,
    label="Cluster Centers",
)
plt.title(f"K-Means Clusters (k={num_clusters}) in 2D PCA Space")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()