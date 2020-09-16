import numpy as np
from scipy import stats

class K_NN:
    
    def __init__(self, k):
        
        """
        :param k: number of nearest neighbours
        """
        self.k = k
    
    def fit(self, data):
        
        """
        :param data: 3D array, where data[i, j] is i-th classes j-th point (vector: D dimenstions)
        """
        n_data = []
        for i in range(data.shape[0]):
            for l in range(data[i].shape[0]):
                n_data.append(np.concatenate((data[i, l], [i])))
        n_data = np.array(n_data)        
        self.old_X = n_data[:,:-1]
        self.old_y = n_data[:,-1]
    
    def predict(self, data):
        
        """
        :param data: 2D array of floats N points each D dimensions
        :return: array of integers
        """
        data = np.array(data)
        shp = data.shape
        if len(data.shape) == 1:
            data = data.reshape([1] + list(data.shape))
        
        # TODO: predict
        prediction = np.zeros(data.shape[0])
        
        for i, r in enumerate(data):
            distance = np.argsort([np.linalg.norm(x-r) for x in self.old_X])[:int(self.k)]
            prediction[i] = np.bincount([self.old_y[i] for i in distance]).argmax()
        
        return np.array(prediction).reshape(shp[:-1])