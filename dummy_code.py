### By this time we are done with all sort of processing, now we will create a K means clustering model and cluster the news vectors

from sklearn.cluster import KMeans

# we also need to select the best value of n_clusters in the clusters
# Following loop will get us the best value of n_clusters

inertia_dict = {}

for k in range(2, 25):
    kmean = KMeans(n_clusters = k)
    kmean.fit(news_vectors)
    inertia_dict[k] = kmean.inertia_


plt.figure(figsize = (12, 9))
sns.lineplot(x = list(inertia_dict.keys()), y = list(inertia_dict.values()))
plt.title('Variation of inertia with number of clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Inertia of the model')
plt.show()


### We cannot find out the value of K using the inertia values since there is no elbow point visible. That is why the next approach will be using Silhouette score.

# Import the Silhouette score function from sklearn

from sklearn.metrics import silhouette_score


silhouette_score_dict = {}

for k in range(2, 25):
    kmean = KMeans(n_clusters = k)
    kmean.fit(news_vectors)
    silhouette_score_dict[k] = silhouette_score(news_vectors, kmean.labels_)


plt.figure(figsize = (12, 9))
sns.lineplot(x = list(silhouette_score_dict.keys()), y = list(silhouette_score_dict.values()))
plt.title('Variation of Silhouette score with number of clusters')
plt.xlabel('Number of clusters')
plt.ylabel('Silhouette score of the model')
plt.show()