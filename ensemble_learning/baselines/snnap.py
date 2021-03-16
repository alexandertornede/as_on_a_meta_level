import logging
import numpy as np
from aslib_scenario.aslib_scenario import ASlibScenario
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, MaxAbsScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.neighbors import BallTree


class SNNAP:

    def __init__(self, clip_runtime=True, feature_selection='chi-squared', top_n=3, k_neighbours=60):
        self._name = 'snnap'
        self._clip_runtime = clip_runtime
        self._feature_selection = feature_selection
        self._top_n = top_n
        self._k_neighbours = k_neighbours
        self._imputer = SimpleImputer()
        self._scaler = MaxAbsScaler()
        self._runtime_scaler = StandardScaler()
        self._models = []
        self._rfr_params = {'n_estimators': 100, 'criterion': 'mse',
                            'max_depth': None, 'min_samples_split': 2}

    def get_name(self):
        return self._name

    def fit(self, scenario: ASlibScenario, fold: int, num_instances: int):
        self._num_algorithms = len(scenario.algorithms)
        self._top_n = min(self._num_algorithms, self._top_n)

        # resample `amount_of_training_instances` instances and preprocess them accordingly
        features, performances = self._resample_instances(
            scenario.feature_data.values, scenario.performance_data.values, num_instances, random_state=fold)
        # TODO: apply feature filtering such as chi-squared based selection technique
        features, performances = self._preprocess_scenario(
            scenario, features, performances)

        # train runtime prediction model for each model
        self._models = [RandomForestRegressor(
            random_state=fold, **self._rfr_params) for alg in range(self._num_algorithms)]
        for num, model in enumerate(self._models):
            model.fit(features, performances[:, num])

        # build index to retrieve k nearest neighbours based on Jaccard distance of best n solvers
        self._index = BallTree(performances, leaf_size=30, metric='pyfunc',
                               func=SNNAP._top_n_jaccard, metric_params={'top_n': self._top_n})
        self._performances = np.copy(performances)

    def predict(self, features, instance_id: int):
        assert(features.ndim == 1), '`features` must be one dimensional'
        features = np.expand_dims(features, axis=0)
        features = self._imputer.transform(features)
        features = self._scaler.transform(features)

        # predict runtimes and get k nearest neighbours based on Jaccard distance of best n solvers
        predicted = np.asarray([model.predict(features)
                                for model in self._models]).reshape(1, -1)
        neighbour_idx = np.squeeze(self._index.query(
            predicted, self._k_neighbours, return_distance=False))

        # find best solver on the instance's k nearest neighbours (best avg. runtime / PAR10 score)
        sub_performances = self._performances[neighbour_idx, :]

        # the summed performance induces a valid ranking
        return np.sum(sub_performances, axis=0)

    def _resample_instances(self, feature_data, performance_data, num_instances, random_state):
        num_instances = min(num_instances, np.size(
            performance_data, axis=0)) if num_instances > 0 else np.size(performance_data, axis=0)
        return resample(feature_data, performance_data, n_samples=num_instances, random_state=random_state)

    def _preprocess_scenario(self, scenario: ASlibScenario, features, performances):
        # TODO: paper does not explicitly mention feature imputation & feature scaling
        features = self._imputer.fit_transform(features)
        features = self._scaler.fit_transform(features)

        # train predictors and select algorithms on running time instead of PAR10 if warranted
        if self._clip_runtime:
            performances = np.clip(
                performances, a_min=np.NINF, a_max=scenario.algorithm_cutoff_time)

        # scale performances to zero mean and unitary standard deviation
        performances = self._runtime_scaler.fit_transform(performances)

        return features, performances

    @staticmethod
    def _top_n_jaccard(x, y, **kwargs):
        top_n = kwargs['metric_params']['top_n']
        top_n_1 = set(np.argpartition(x, top_n)[:top_n])
        top_n_2 = set(np.argpartition(y, top_n)[:top_n])

        return len(top_n_1.intersection(top_n_2)) / float(len(top_n_1.union(top_n_2)))
