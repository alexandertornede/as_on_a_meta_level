import logging
import random
import numpy as np
from aslib_scenario.aslib_scenario import ASlibScenario
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from baselines.gmeans import GMeans


class ISAC:

    def __init__(self):
        self._name = 'isac'
        self._imputer = SimpleImputer()
        self._scaler = MaxAbsScaler()
        self._solvers = {}

    def get_name(self):
        return self._name

    def fit(self, scenario: ASlibScenario, fold: int, num_instances: int):
        self._num_algorithms = len(scenario.algorithms)
        self._algorithm_cutoff_time = scenario.algorithm_cutoff_time

        # resample `amount_of_training_instances` instances and preprocess them accordingly
        features, performances = self._resample_instances(
            scenario.feature_data.values, scenario.performance_data.values, num_instances, random_state=fold)
        features, performances = self._preprocess_scenario(
            scenario, features, performances)

        # fit g-means clustering on normalized feature vectors and compute respectively best solvers
        self._gmeans = GMeans(random_state=fold).fit(features)
        for label, center in enumerate(self._gmeans.cluster_centers_):
            performances_ = performances[self._gmeans.labels_ == label]
            self._solvers[label] = np.argmin(np.nanmean(performances_, axis=0))

        # compute SBS which will be used for instances deviating too harshly from every computed centroid
        self._sbs = np.argmin(np.sum(performances, axis=0))    
        self._threshold = self._compute_threshold(features, self._gmeans.cluster_centers_, self._gmeans.labels_)


    def predict(self, features, instance_id: int):
        assert(features.ndim == 1), '`features` must be one dimensional'
        features = np.expand_dims(features, axis=0)
        features = self._imputer.transform(features)
        features = self._scaler.transform(features)

        # compute nearest cluster and determine whether the distance exceeds the threshold
        label = self._gmeans.predict(features)[0]
        dist = np.linalg.norm(self._gmeans.cluster_centers_[label] - features, ord=2)
        
        # return ranking that sets the predicted solver to 0, any other solver to 1
        ranking = np.ones(self._num_algorithms)
        if dist > self._threshold:
            ranking[self._sbs] = 0
        else:
            ranking[self._solvers[label]] = 0

        return ranking

    def _resample_instances(self, feature_data, performance_data, num_instances, random_state):
        num_instances = min(num_instances, np.size(performance_data, axis=0)) if num_instances > 0 else np.size(performance_data, axis=0)
        return resample(feature_data, performance_data, n_samples=num_instances, random_state=random_state)

    def _preprocess_scenario(self, scenario, features, performances):
        features = self._imputer.fit_transform(features)
        features = self._scaler.fit_transform(features)

        return features, performances

    @staticmethod
    def _compute_threshold(X, cluster_centers, labels):
        tmp = np.asarray([])

        for label, center in enumerate(cluster_centers):
            X_ = X[labels == label]
            dist = np.linalg.norm(X_ - center, axis=1, ord=2) 
            tmp = np.hstack([tmp, dist])
        
        return np.average(tmp) + 2*np.std(tmp)
