#coding:utf-8
import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

X,y = make_blobs(n_samples=150, n_features=2, centers=3, cluster_std=0.5, shuffle=True, random_state=0)

print X, len(X), type(X)
# plt.scatter(X[:,0], X[:,1], c='white', marker='o', s=50)
# plt.grid()
# plt.show()

# km = KMeans(n_clusters=3, init='random', n_init=10, max_iter=300, tol=1e-04, random_state=0)
# y_km = km.fit_predict(X)
# print ('Distortion: %.2f' % km.inertia_)

# plt.scatter(X[y_km==0,0], X[y_km==0,1], s=50, c='lightgreen', marker='s', label='cluster 1')
# plt.scatter(X[y_km==1,0], X[y_km==1,1], s=50, c='orange', marker='o', label='cluster 2')
# plt.scatter(X[y_km==2,0], X[y_km==2,1], s=50, c='lightblue', marker='v', label='cluster 3')
# plt.scatter(km.cluster_centers_[:,0], km.cluster_centers_[:,1], s=250, marker='*', c='red', label='centroids')
# plt.legend()
# plt.grid()
# plt.show()

# distortions = []
# for i in range(1,11):
#     km = KMeans(n_clusters=i, init='k-means++', n_init=10, max_iter=300, random_state=0)
#     km.fit(X)
#     distortions.append(km.inertia_)
#
# plt.plot(range(1,11), distortions, marker='o')
# plt.xlabel("Number of clusters")
# plt.ylabel('Distortion')
# plt.show()

km = KMeans(n_clusters=3, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km = km.fit_predict(X)
cluster_labels = np.unique(y_km)
n_clusters = cluster_labels.shape[0]
silhouette_vals = silhouette_samples(X, y_km, metric='euclidean')  #所有数据点的剪影系数
y_ax_lower, y_ax_upper = 0, 0
yticks = []
for i, c in enumerate(cluster_labels):
    c_silhouette_vals = silhouette_vals[y_km == c]  #某个类别的剪影系数
    c_silhouette_vals.sort()
    y_ax_upper += len(c_silhouette_vals)  #计算
    color = cm.jet(i/n_clusters)
    print c_silhouette_vals
    plt.barh(range(y_ax_lower, y_ax_upper), c_silhouette_vals, height=1.0, edgecolor='None', color=color)
    yticks.append((y_ax_lower+y_ax_upper)/2)
    y_ax_lower += len(c_silhouette_vals)
silhouette_avg = np.mean(silhouette_vals)
plt.axvline(silhouette_avg, color='red', linestyle='--')
plt.yticks(yticks, cluster_labels+1)
plt.ylabel('Cluster')
plt.xlabel('silhouette coefficient')
plt.show()

km = KMeans(n_clusters=2, init='k-means++', n_init=10, max_iter=300, tol=1e-04, random_state=0)
y_km = km.fit_predict(X)
