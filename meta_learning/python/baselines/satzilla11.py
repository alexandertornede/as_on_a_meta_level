import logging
import random
import numpy as np
from aslib_scenario.aslib_scenario import ASlibScenario
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from collections import Counter


class SATzilla11:
    """ Implementation of SATzilla's internal algorithm selector. 

    Note, however, that this implementation does not account for SATzilla's many other steps, 
    thus does not account for presolving or employing a backup solver.    
    """

    def __init__(self):
        self._name = 'satzilla-11'
        self._models = {}
        self._imputer = SimpleImputer()
        self._scaler = StandardScaler()
        # used to break ties during the predictions phase
        self._rand = random.Random(0)

    def get_name(self):
        return self._name

    def fit(self, scenario: ASlibScenario, fold: int, num_instances: int):
        # TODO: assert that for all given features were computed within the threshold?
        self._num_algorithms = len(scenario.algorithms)
        self._algorithm_cutoff_time = scenario.algorithm_cutoff_time

        # resample `amount_of_training_instances` instances and preprocess them accordingly
        features, performances = self._resample_instances(
            scenario.feature_data.values, scenario.performance_data.values, num_instances, random_state=fold)
        features, performances = self._preprocess_scenario(
            scenario, features, performances)

        # create and fit rfcs' for all pairwise comparisons between two algorithms
        self._pairwise_indices = [(i, j) for i in range(
            self._num_algorithms) for j in range(i + 1, self._num_algorithms)]

        for (i, j) in self._pairwise_indices:
            # determine pairwise target, initialize models and fit each RFC wrt. instance weights
            pair_target = self._get_pairwise_target((i, j), performances)
            sample_weights = self._compute_sample_weights(
                (i, j), performances)

            # account for nan values (i.e. ignore pairwise comparisons that involve an algorithm run violating
            # the cutoff if the config specifies 'ignore_censored'), hence set all respective weights to 0
            sample_weights = np.nan_to_num(sample_weights)

            # TODO: how to set the remaining hyperparameters? 
            self._models[(i, j)] = RandomForestClassifier(
                n_estimators=99, max_features='log2', n_jobs=1, random_state=fold)
            self._models[(i, j)].fit(features, pair_target,
                                     sample_weight=sample_weights)

    def predict(self, features, instance_id: int):
        assert(features.ndim == 1), '`features` must be one dimensional'
        features = np.expand_dims(features, axis=0)
        features = self._imputer.transform(features)
        features = self._scaler.transform(features)

        # compute the most voted algorithms for the given instance
        predictions = {(i, j): rfc.predict(features).item()
                       for (i, j), rfc in self._models.items()}

        counter = Counter(predictions.values())
        max_votes = max(counter.values())
        most_voted = set(alg for alg, votes in counter.items()
                         if votes >= max_votes)

        # break ties according to solely focussing on votes of respective models that involve
        # only algorithms with most received votes
        if len(most_voted) > 1:
            indices = [(i, j) for (
                i, j) in self._pairwise_indices if i in most_voted and j in most_voted]
            predictions = {(i, j): predictions[(i, j)] for (i, j) in indices}
            counter = Counter(predictions.values())
            max_votes = max(counter.values())
            most_voted = set(
                alg for alg, votes in counter.items() if votes >= max_votes)

            # break remaining ties in random fashion
            if len(most_voted) > 1:
                selection = self._rand.choice(list(most_voted))

            else:
                selection = next(iter(most_voted))

        else:
            selection = next(iter(most_voted))

        # create ranking st. the selected algorithm has rank 0, any other algorithm has rank 1
        ranking = np.ones(self._num_algorithms)
        ranking[selection] = 0
        return ranking

    def _resample_instances(self, feature_data, performance_data, num_instances, random_state):
        num_instances = min(num_instances, np.size(performance_data, axis=0)) if num_instances > 0 else np.size(performance_data, axis=0)
        return resample(feature_data, performance_data, n_samples=num_instances, random_state=random_state)

    def _preprocess_scenario(self, scenario, features, performances):
        features = self._imputer.fit_transform(features)
        features = self._scaler.fit_transform(features)

        return features, performances

    def _get_pairwise_target(self, pair, performances):
        i, j = pair

        # label target as i if the runtime of algorithm i is <= the runtime of algorithm j, otherwise label it j
        pair_target = np.full(performances.shape[0], fill_value=i)
        pair_target[performances[:, i] > performances[:, j]] = j

        return pair_target

    def _compute_sample_weights(self, pair, performances):
        i, j = pair
        weights = np.absolute(performances[:, i] - performances[:, j])

        return weights
