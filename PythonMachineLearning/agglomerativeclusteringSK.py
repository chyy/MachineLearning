#coding:utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist,squareform
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering

np.random.seed(123)
variables = ['X', 'Y', 'Z']
labels = ['ID_0', 'ID_1', 'ID_2', 'ID_3', 'ID_4']
X = np.random.random_sample([5,3])*10
# df = pd.DataFrame(X, columns=variables, index=labels)
#
# row_dist = pd.DataFrame(squareform(pdist(df, metric='euclidean')), columns=labels, index=labels)
# #print row_dist
# #help(linkage)
#
# row_clusters = linkage(row_dist, method='complete')
# result = pd.DataFrame(row_clusters,
#              columns=['row label 1', 'row label 2', 'distance', 'no. of items in clust.'],
#              index=['cluster %d' %(i+1) for i in range(row_clusters.shape[0])])
#
# row_dendr = dendrogram(row_clusters, labels=labels)
# plt.tight_layout()
# plt.ylabel('Euclidean distance')
# plt.show()

ac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='complete')
label = ac.fit_predict(X)
print ('Cluster labels: %s' % label)