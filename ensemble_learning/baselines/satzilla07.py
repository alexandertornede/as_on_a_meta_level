import numpy as np
from .utils import impute_censored
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, make_scorer
from aslib_scenario.aslib_scenario import ASlibScenario
from mlxtend.feature_selection import SequentialFeatureSelector
from .utils import distr_func


class SATzilla07:
    def __init__(self):
        self._name = 'satzilla-07'
        self._models = {}
        self._features = {}
        self._quad_features = {}
        self._imputer = SimpleImputer()
        self._scaler = StandardScaler()

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

        base_model = Ridge(alpha=1.0, random_state=fold)
        scorer = make_scorer(mean_squared_error, greater_is_better=False)
        sfs_params = {'estimator': base_model, 'k_features': 'best',
                          'forward': True, 'scoring': scorer, 'cv': 2}

        for num in range(self._num_algorithms):
            feature_selector = SequentialFeatureSelector(**sfs_params)
            feature_selector = feature_selector.fit(
                features, performances[:, num])
            self._features[num] = feature_selector.k_feature_idx_

            features_tmp = PolynomialFeatures(2).fit_transform(
                features[:, self._features[num]])

            feature_selector = SequentialFeatureSelector(**sfs_params)
            feature_selector = feature_selector.fit(
                features_tmp, performances[:, num])
            self._quad_features[num] = feature_selector.k_feature_idx_
            features_tmp = features_tmp[:, self._quad_features[num]]

            censored = performances[:, num] >= self._algorithm_cutoff_time
            self._models[num] = impute_censored(
                features_tmp, performances[:, num], censored, base_model, distr_func, self._algorithm_cutoff_time)

    def predict(self, features):
        assert(features.ndim == 1), '`features` must be one dimensional'
        features = np.expand_dims(features, axis=0)
        features = self._imputer.transform(features)
        features = self._scaler.transform(features)

        predictions = np.empty(self._num_algorithms)
        for num in range(self._num_algorithms):
            features_tmp = features[self._features[num]]
            features_tmp = PolynomialFeatures(2).fit_transform(
                features[:, self._features[num]])
            predictions[num] = self._model[num].predict(
                features_tmp[self._quad_features[num]])

        return np.argmin(predictions)

    def _resample_instances(self, feature_data, performance_data, num_instances, random_state):
        num_instances = min(num_instances, np.size(
            performance_data, axis=0)) if num_instances > 0 else np.size(performance_data, axis=0)
        return resample(feature_data, performance_data, n_samples=num_instances, random_state=random_state)

    def _preprocess_scenario(self, scenario, features, performances):
        features = self._imputer.fit_transform(features)
        features = self._scaler.fit_transform(features)

        return features, performances
