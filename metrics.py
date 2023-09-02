import numpy as np

class Metric:
    def __init__(self):
        pass

    def value(self, monitoring_data):
        num_samples = np.shape(monitoring_data)[0]
        return np.zeros(num_samples) 

class Identity(Metric):
    def __init__(self, key):
        self.key = key

    def value(self, monitoring_data):
        return monitoring_data[self.key]

class Variance(Metric):
    def __init__(self, key="fcov"):
        self.key = key

    def value(self, monitoring_data):
        detection_variable = monitoring_data[self.key]
        result = np.maximum(detection_variable, axis=1)    
        return result
        
class CovarianceWeightedAverage(Metric):
    def __init__(self, timesteps, window_size=30, key="fcov", sigma=2.5):
        self.timesteps = timesteps
        self.window_size = window_size
        self.key = key
        self.sigma = sigma

    def _gaussian_window(self, axis=1):
        t = np.expand_dims(np.linspace(-self.window_size, self.window_size, self.timesteps), axis=0)
        g = np.exp(-np.power(t, 2.0) / (2 * np.power(self.sigma, 2.0)))
        return g / np.sum(g, axis=axis, keepdims=True)

    def value(self, monitoring_data):
        detection_variable = monitoring_data[self.key]
        g = self._gaussian_window()
        result = np.maximum(np.sum(detection_variable * g, axis=1), 0)    
        return result