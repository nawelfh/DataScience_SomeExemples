import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from matplotlib.colors import ListedColormap
dataset=ps=pd.read_csv('Mall_Customers.csv')
X=dataset.iloc[:,[2,3]].values

#wcss=[]
#for i in range(1,11):
#    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=42)
#    kmeans.fit(X)
#    wcss.append(kmeans.inertia_)
#plt.plot(range(1,11),wcss)
#plt.title('The Elbow Method')
#plt.xlabel('Number of clusters')
#plt.ylabel('WCSS')
#plt.show()

kmeans=KMeans(n_clusters=10,random_state=42)
kmeans.fit(X)
cluster_centers=kmeans.cluster_centers_
y_set=kmeans.labels_

for i, j in enumerate(np.unique(y_set)):
      plt.scatter(X[y_set == j, 0], X[y_set == j, 1],
                          c = ListedColormap(('red', 'green','orange','purple','black','brown','magenta','pink','gray','cyan','yellow'))(i), label = j)
plt.scatter( cluster_centers[0,0], cluster_centers[0,1],s=150,color='gray',marker='s')
plt.scatter(cluster_centers[1,0], cluster_centers[1,1],s=150,color='blue',marker='s')
plt.scatter(cluster_centers[2,0], cluster_centers[2,1],s=150,color='brown',marker='s')
plt.scatter(cluster_centers[3,0], cluster_centers[3,1],s=150,color='black',marker='s')
plt.scatter(cluster_centers[4,0], cluster_centers[4,1],s=150,color='magenta',marker='s')
plt.scatter(cluster_centers[5,0], cluster_centers[5,1],s=150,color='magenta',marker='s')
plt.scatter(cluster_centers[6,0], cluster_centers[6,1],s=150,color='magenta',marker='s')
plt.scatter(cluster_centers[7,0], cluster_centers[7,1],s=150,color='magenta',marker='s')
plt.scatter(cluster_centers[8,0], cluster_centers[8,1],s=150,color='magenta',marker='s')
plt.scatter(cluster_centers[9,0], cluster_centers[9,1],s=150,color='magenta',marker='s')

X1, X2 = np.meshgrid(np.arange(start = X[:, 0].min() - 1, stop = X[:, 0].max() + 1, step = 0.01),
               np.arange(start = X[:, 1].min() - 1, stop = X[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, kmeans.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                       alpha = 0.4, cmap = ListedColormap(('red', 'green','orange','purple','black','brown','magenta','pink','gray','cyan','yellow')))
plt.show()