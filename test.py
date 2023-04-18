# import matplotlib.pyplot as plt
# from sklearn.datasets import make_blobs
# from sklearn.preprocessing import StandardScaler
# from unsupervised.kmeans import KMeans

# # Specifying the number of cluster our data should have
# n_components = 4

# X, true_labels = make_blobs(
#     n_samples=750, centers=n_components, cluster_std=0.4, random_state=0
# )

# plt.title("Unclustered Data")
# plt.scatter(X[:, 0], X[:, 1], s=15)
# plt.xticks([])
# plt.yticks([])
# plt.show()


# # Initialize KMeans
# kmeans = KMeans(n_clusters=4)

# # fit the data & predict cluster labels
# kmeans.fit(X)
# predicted_labels = kmeans.predict(X)


# # Based on predicted_labels, we assign each data point distinct colour
# colors = ["#4EACC5", "#FF9C34", "#4E9A06", "m"]
# for k, col in enumerate(colors):
#     cluster_data = predicted_labels == k
#     plt.scatter(X[cluster_data, 0], X[cluster_data, 1], s=15)

    
# plt.title("Clustered Data")
# plt.xticks([])
# plt.yticks([])
# plt.show()
import math
import numpy as np

x=[12,28]
y=[15,38]

total=0
for i in range(len(x)):
    a=(y[i]-x[i])**2
    total+=a
    
print(round(math.sqrt(total),8))

x=np.array(x)
y=np.array(y)
print(np.linalg.norm(x-y))

x=[[12,28]]
y=[[15,38]]
from sklearn.metrics.pairwise import euclidean_distances
print(euclidean_distances(x,y))





    