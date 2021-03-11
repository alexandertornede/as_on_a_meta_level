import warnings
import numpy as np
from sklearn.cluster import KMeans
from scipy.stats import anderson
from sklearn.utils import check_random_state
from sklearn.metrics import pairwise_distances


class GMeans:
    def __init__(self, min_samples=0.001, significance=0.05, n_init=5, n_init_kmeans=5, n_init_final=5, random_state=None):
        self._min_samples = min_samples   
        self._significance = [0.15, 0.1, 0.05, 0.025, 0.001].index(significance)
        self._n_init = n_init
        self._n_init_kmeans = n_init_kmeans 
        self._n_init_final = n_init_final
        # self._random_state = check_random_state(random_state)
        self._random_state = random_state
        self._kmeans = None


    def fit(self, X):
        self.inertia_ = np.inf
        # compute absolute quantity of `min_samples` from fraction
        if self._min_samples < 1.0:
            self._min_samples = X.shape[0] * self._min_samples
        
        for _ in range(self._n_init):
            kmeans = KMeans(n_clusters=1, n_init=1, random_state=self._random_state).fit(X)
            queue = [0]

            while queue:
                center_idx = queue.pop()

                # get instances assigned to the current cluster center
                X_ = X[kmeans.labels_ == center_idx]
                if np.size(X_, axis=0) <= 2:
                    continue

                _, counts = np.unique(kmeans.labels_, return_counts=True)
                counts, count_counts = np.unique(counts, return_counts=True)

                # fit kmeans with two centroids on all instances assigned to the current cluster
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    tmp_kmeans = KMeans(n_clusters=2, n_init=self._n_init_kmeans, random_state=self._random_state).fit(X_) 
                
                # check whether the determined (sub-)clustering follows a Gaussian distribution when
                # projected on the vector that connects the two child-centroids
                child_one, child_two = tmp_kmeans.cluster_centers_
                
                v = child_two - child_one
                tmp_labels = tmp_kmeans.predict(X)
                unique = np.unique(tmp_labels[kmeans.labels_ == center_idx])

                if np.linalg.norm(v, ord=2) <= 0.0 or unique.size < 2:
                    continue
                
                # normalize to unit variance and compute Anderson statistics
                y = np.inner(v, X_) / np.linalg.norm(v, ord=2)
                mean = np.mean(y)
                std = np.std(y)
                y = (y - mean) / std
                A2, critical, sig = anderson(y)

                if A2 > critical[self._significance]:
                    # replace old cluster center                
                    kmeans.cluster_centers_ = np.delete(kmeans.cluster_centers_, center_idx, axis=0)
                    kmeans.cluster_centers_ = np.vstack([kmeans.cluster_centers_, child_one, child_two])
                    offset = np.size(kmeans.cluster_centers_, axis=0) - 2
                    
                    # maintain cluster labels (assignment must be updated due to deletion and insertion of new centers)
                    del_idx = kmeans.labels_ > center_idx
                    ins_idx = kmeans.labels_ == center_idx
                    kmeans.labels_[del_idx] -= 1
                    kmeans.labels_[ins_idx] = tmp_labels[ins_idx] + offset

                    # add children to queue for further processing
                    queue.extend([offset, offset + 1])
            
            if kmeans.inertia_ < self.inertia_:
                self.inertia_ = kmeans.inertia_
                self._k = np.size(kmeans.cluster_centers_, axis=0)
                
            self._kmeans = KMeans(n_clusters=self._k, n_init=self._n_init_final, random_state=self._random_state).fit(X)
            self.inertia_ = self._kmeans.inertia_  
            self.cluster_centers_ = self._kmeans.cluster_centers_
            self.labels_ = self._kmeans.labels_

        return self


    def predict(self, X):
        return self._kmeans.predict(X)

    def fit_predict(self, X):
        self.fit(X)
        return self._kmeans.predict(X)

    def _redistribute(self, X):
        redistribute = {label: center for label, center in enumerate(self._kmeans.cluster_centers_)}
        
        while redistribute:
            label, center = redistribute.popitem()
            # get instances assigned to the current cluster center
            X_ = X[self._kmeans.labels_ == label]
            
            if np.size(X_, axis=0) >= self._min_samples:
                continue
            
            # compute the second nearest cluster center for each instance respectively
            distances = pairwise_distances(X_, self._kmeans.cluster_centers_, metric='euclidean')
            assignments = np.argpartition(distances, 1, axis=1)[:, 1]
     
            # assign instances to new cluster centers 
            self._kmeans.labels_[self._kmeans.labels_ == label] = assignments
            self._kmeans.cluster_centers_ = np.delete(self._kmeans.cluster_centers_, label, axis=0)
            self._kmeans.labels_[self._kmeans.labels_ > label] -= 1
            self.labels_ = self._kmeans.labels_
            self.cluster_centers_ = self._kmeans.cluster_centers_
            
            # recompute centroid of all clusters whom instances have been assigned
            centroids = np.zeros(self.cluster_centers_.shape)
            for label, center in enumerate(self.cluster_centers_):
                centroids[label] = np.mean(X[self.labels_ == label], axis=0)

            self.cluster_centers_ = centroids

