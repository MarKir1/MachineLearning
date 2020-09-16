import numpy as np

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
        pass
    
class KMeansPlusPlus(KMeans):
    def L2_norm(vector):
        count = 0
        for i in vector:
            i = i**2
            count += i
        norm = np.sqrt(count)
        return norm
    
    def _initialize_means(self, data):
        # TODO: Initialize cluster centers using K Means++ algorithm
        idcs = list(range(data.shape[0]))
        idx = np.random.randint(0, data.shape[0], size = 1)
        idcs.remove(idx)
        means = []
        means.append(data[idx][0])

        for clusters in range(self.k-1):

            distances = []
            for i in idcs:
                distance_i = 0
                for mean in means:
                    distance_i += KMeansPlusPlus.L2_norm(data[i] - mean[0])
                distances.append([data[i], distance_i])

            count = 0
            for i in distances:
                count += i[1]

            for i in range(len(distances)):
                distances[i][1] = distances[i][1]/count

            for i in range(len(distances)):
                if i == 0:
                    distances[i][1] = distances[i][1]
                else:
                    distances[i][1] = distances[i-1][1] + distances[i][1]

            rand = np.random.uniform(0,1, size = 1)

            for i in range(len(distances)):
                if np.all(rand < distances[i][1]):
                    choice = i-1
                    break

            new_mean = np.array(data[choice])
            means.append(list(new_mean))
        
        self.means = means

        return self.means