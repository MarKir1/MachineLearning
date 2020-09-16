import numpy as np
import random
import math
import scipy.stats


class GMM:
    def __init__(self, k):
        self.k = k
        self.means = []
        self.covariances = []
        self.pis = []
        self.gammas = []
        self.tolerance = 1e-6

    def fit(self, data):
        """
        :params data: np.array of shape (..., dim)
                                  where dim is number of dimensions of point
        """
        self._initialize_params(data)
        self.data = np.array(data)
        new_lld = self.loss(self.data)
        recursion = 0
        lld = 0
        while recursion < 500 and np.abs(new_lld - lld) > self.tolerance:
            print(recursion, 'GMM')
            lld = new_lld
            self._E_step(data)
            self._M_step(data)
            new_lld = self.loss(data)
            recursion += 1
        self.means = np.array(self.means)
        self.covariances = np.array(self.covariances)
        self.pis = np.array(self.pis)

    def _initialize_params(self, data):
    # TODO: initialize means, covariances, pis
        classifiers = KMeans(self.k)
        classifiers.fit(data)
        self.means = classifiers.means
        self.pis = np.ones(self.k) / self.k
        self.covariances = np.array([1e9 * np.eye(data.shape[1]) for i in range(self.k)])

    def _E_step(self, data):
    # TODO: find gammas from means, covariances, pis
        self.data = np.array(data)
        n, _ = data.shape
        self.gammas = np.array([float(self.pis[k]) * self.Normal(data, self.means[k], self.covariances[k]) for k in range(self.k)]).T
        self.gammas = np.array([i / i.sum() for i in self.gammas]).T
        return self.gammas

    def _M_step(self, data):
    # TODO: find means, covariances, pis from gammas
        self.pis = np.sum(self.gammas, axis=0) / len(data)

        for k in range(self.k):
            means = np.array(self.means)
            total = np.einsum('ij, i', self.data,
            self.gammas[k], optimize=True)
            self.means[k] = total / np.sum(self.gammas[k])

        for k in range(self.k):
            data = self.data - self.means[k]
            summ = np.einsum('i, ij, ik', self.gammas[k], data, data, optimize=True)
            self.covariances[k] = summ / np.sum(self.gammas[k])

        return self.pis, self.means, self.covariances

    def predict(self, data):
        """
        :param data: np.array of shape (..., dim)
        :return: np.array of shape (...) without dims
        each element is integer from 0 to k-1
        """
        expectations = _E_step(data, self.pis, self.means, self.covariance)
        return np.argmax(expectations, axis=1)

    def get_means(self):
        return self.means.copy()

    def get_covariances(self):
        return self.covariances.copy()

    def get_pis(self):
        return self.pis.copy()

    def Normal(self, Xi, Mk, Ck):
    # Calculate the value for Xi in normal distribution k
    # Xi - data[i]
    # Mk - means[k]
    # Ck - covariances[k]
        probability = scipy.stats.multivariate_normal.pdf(Xi, Mk, Ck)
        return probability

    def loss(self, data):
        new_lld = 0
        probs = np.array([float(self.pis[k]) * self.Normal(data, self.means[k], self.covariances[k]) for k in range(self.k)]).T 
        return np.sum(np.log(np.sum(probs[i])) for i in range(len(data)))
        for i in range(len(data)):
            temporary = 0
            for k in range(self.k):
                temporary += self.pis[k] * \
                    self.Normal(data[i], self.means[k], self.covariances[k])
            new_lld += np.log(temporary)
        return new_lld


# ---------------------------------------------------------------------------------------------------------------
class KMeans():
    def __init__(self, k, maximum_iterations = 900, tolerance = 0.01): # need 'tolerance' and 'max. iterations'?
        self.k = k
        self.means = None # Means_of_Clusters
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance

    def _initialize_means(self, data):
        self.means = data[np.random.randint(0, high=len(data), size=self.k)]
    
    def fit(self, data):
        """
        :param data: numpy array of shape (k, ..., dims)
        """
        self.dim = data.shape[-1]
        self._initialize_means(data)
        
        # TODO: Initialize Mixtures, then run EM algorithm until it converges.
        
        if len(data.shape) >= 3:
            data = data.reshape(-1, self.dim)
        
        # Start_of_iterations
        for i in range(self.maximum_iterations):
            self.clusters = {}
            for i in range(self.k):
                self.clusters[i] = []
                
        # Finding_point_and_cluster_distance,_choice_of_the_nearest_mean
        
            for i in data:
                distances = [np.linalg.norm(i - mean) for mean in self.means]
                cluster_choice = distances.index(min(distances))
                self.clusters.setdefault(cluster_choice,[])
                self.clusters[cluster_choice].append(i)

            old = self.means
            
        # Re-calculating_the_centroids_to_find_cluster datapoints'_average
            
            self.means = []
            
            for cluster_choice in self.clusters:
                mean = np.average(np.array(self.clusters[cluster_choice]), axis = 0)
                self.means.append(mean)        
            
            Optimal = True
            
            for i, mean in enumerate(self.means):
                old_mean = old[i]
                new_mean = mean
                               
                if np.sum(100.0 * (new_mean - old_mean) / old_mean) > self.tolerance:
                    Optimal = False
                    
        # Break_if_optimal_results
            if Optimal:
                break
    
    #def _initialize_means(self, data):
        # TODO: Initialize cluster centers

    def predict(self, data):
        """
        :param data: numpy array of shape (k, ..., dims)
        :return: labels of each datapoint and it's mean
                 0 <= labels[i] <= k - 1
        """
        cluster_choice = []
        means = []
        
        if len(data.shape) >= 3:
            data = data.reshape(-1, self.dim)
        
        for i in data:
            distances = [np.linalg.norm(i - mean) for mean in self.means]
            cluster_choice1 = distances.index(min(distances))
            cluster_choice.append(cluster_choice1)
            means.append(self.means[cluster_choice1])
        return np.array(cluster_choice), np.array(means)