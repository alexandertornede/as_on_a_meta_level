import logging
import numpy as np
from itertools import chain, combinations
from aslib_scenario.aslib_scenario import ASlibScenario
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.neighbors import KDTree


class SUNNY:
    def __init__(self, determine_best='min-par10'):
        self._name = 'sunny'
        self._imputer = SimpleImputer(missing_values=np.nan, strategy='mean')
        self._scaler = StandardScaler()
        self._determine_best = determine_best
        self._k = 16

    def get_name(self):
        return self._name

    def fit(self, scenario: ASlibScenario, fold: int, num_instances: int):
        self._num_algorithms = len(scenario.algorithms)
        self._algorithm_cutoff_time = scenario.algorithm_cutoff_time

        # resample `amount_of_training_instances` instances and preprocess them accordingly
        features, performances = self._resample_instances(
            scenario.feature_data, scenario.performance_data, num_instances, random_state=fold)
        features, performances = self._preprocess_scenario(
            scenario, features, performances)

        # build nearest neighbors index based on euclidean distance
        self._model = KDTree(features, leaf_size=30, metric='euclidean')
        self._performances = np.copy(performances)

    def predict(self, features, instance_id: int):
        assert(features.ndim == 1), '`features` must be one dimensional'
        features = np.expand_dims(features, axis=0)
        features = self._imputer.transform(features)
        features = self._scaler.transform(features)

        neighbour_idx = np.squeeze(self._model.query(
            features, k=self._k, return_distance=False))

        if self._determine_best == 'subportfolio':
            if np.isnan(self._performances).any():
                raise NotImplementedError()

            sub_portfolio = self._build_subportfolio(neighbour_idx)
            schedule = self._build_schedule(neighbour_idx, sub_portfolio)
            selection = schedule[0]

        elif self._determine_best == 'max-solved':
            if np.isnan(self._performances).any():
                raise NotImplementedError()
            
            # select the algorithm which solved the most instances (use min PAR10 as tie-breaker)
            sub_performances = self._performances[neighbour_idx, :]
            num_solved = np.sum(sub_performances <
                                self._algorithm_cutoff_time, axis=0)
            max_solved = np.max(num_solved)
            indices, = np.where(num_solved >= max_solved)
            sub_performances = sub_performances[:, indices]
            runtime = np.sum(sub_performances, axis=0)
            selection = indices[np.argmin(runtime)]

        elif self._determine_best == 'min-par10':
            # select the algorithm with the lowest mean PAR10 score (use max solved as tie-breaker)
            sub_performances = self._performances[neighbour_idx, :]
            runtime = np.nanmean(sub_performances, axis=0)
            
            if not np.isnan(runtime).all():
                min_runtime = np.nanmin(runtime)
                runtime = np.nan_to_num(runtime, nan=np.inf)

            else:
                return np.random.choice(self._num_algorithms)

            indices, = np.where(runtime <= min_runtime)
            sub_performances = sub_performances[:, indices]

            num_solved = np.sum(np.nan_to_num(sub_performances, nan=np.inf) < self._algorithm_cutoff_time)
            selection = indices[np.argmax(num_solved)]

        else:
            ValueError('`{}` is no valid selection strategy'.format(
                self._determine_best))

        # create ranking st. the selected algorithm has rank 0, any other algorithm has rank 1
        ranking = np.ones(self._num_algorithms)
        ranking[selection] = 0
        return ranking

    def _build_subportfolio(self, neighbour_idx):
        sub_performances = self._performances[neighbour_idx, :]

        # naive, inefficient computation
        algorithms = range(self._num_algorithms)
        num_solved, avg_time = np.NINF, np.NINF
        sub_portfolio = None
        for subset in chain.from_iterable(combinations(algorithms, n) for n in range(1, len(algorithms))):
            # compute number of solved instances and average solving time
            tmp_solved = np.count_nonzero(
                np.min(sub_performances[:, subset], axis=1) < self._algorithm_cutoff_time)

            # TODO: not entirely sure whether this is the correct way to compute the average runtime as mentioned in the paper
            tmp_avg_time = np.sum(
                sub_performances[:, subset]) / sub_performances[:, subset].size
            if tmp_solved > num_solved or (tmp_solved == num_solved and tmp_avg_time < avg_time):
                num_solved, avg_time = tmp_solved, tmp_avg_time
                sub_portfolio = subset

        return sub_portfolio

    def _build_schedule(self, neighbour_idx, sub_portfolio):
        # schedule algorithms wrt. to solved instances (asc.) and break ties according to its average runtime (desc.)
        sub_performances = self._performances[neighbour_idx, :]
        alg_performances = {alg: (np.count_nonzero(
            sub_performances[:, alg] < self._algorithm_cutoff_time), (-1)*np.sum(sub_performances[:, alg])) for alg in sub_portfolio}
        schedule = sorted([(solved, avg_time, alg) for (
            alg, (solved, avg_time)) in alg_performances.items()], reverse=True)

        return [alg for (_, _, alg) in schedule]

    def _resample_instances(self, feature_data, performance_data, num_instances, random_state):
        num_instances = min(num_instances, np.size(
            performance_data, axis=0)) if num_instances > 0 else np.size(performance_data, axis=0)
        return resample(feature_data, performance_data, n_samples=num_instances, random_state=random_state)

    def _preprocess_scenario(self, scenario, features, performances):
        features = self._imputer.fit_transform(features)
        features = self._scaler.fit_transform(features)

        return features, performances
