import numpy as np


class Metric:
    def __init__(self):
        pass

    def value(self, monitoring_data):
        num_samples = np.shape(monitoring_data)[0]
        return np.zeros(num_samples)


class Identity(Metric):
    def __init__(self, monitored_params=[]):
        self.monitored_params = monitored_params

    def value(self, monitoring_data):
        return monitoring_data[self.monitored_params[0]]

    @property
    def monitored_params(self):
        return [self.input_param]


class Variance(Metric):
    def __init__(self):
        pass

    def value(self, monitoring_data):
        detection_variable = monitoring_data[self.INPUT_PARAM]
        result = np.maximum(detection_variable, axis=1)
        return result

    @property
    def monitored_params(self):
        return [self.input_param]


class CovarianceWeightedAverage(Metric):
    def __init__(self, input_param, window_size=30, sigma=2.5):
        self.input_param = input_param
        self.window_size = window_size
        self.sigma = sigma

    def _gaussian_window(self, timesteps, axis=1):
        t = np.expand_dims(
            np.linspace(-self.window_size, self.window_size, timesteps), axis=0
        )
        g = np.exp(-np.power(t, 2.0) / (2 * np.power(self.sigma, 2.0)))
        return g / np.sum(g, axis=axis, keepdims=True)

    def value(self, monitoring_data):
        detection_variable = monitoring_data[self.input_param]
        n_timesteps = np.shape(detection_variable)[1]
        g = self._gaussian_window(n_timesteps)
        result = np.maximum(np.sum(detection_variable * g, axis=1), 0)
        return result

    @property
    def monitored_params(self):
        return [self.input_param]
