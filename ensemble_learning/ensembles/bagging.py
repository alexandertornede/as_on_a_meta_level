import copy
import logging
import numpy as np
import sys

from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from aslib_scenario.aslib_scenario import ASlibScenario

from ensembles.prediction import predict_with_ranking
from ensembles.validation import base_learner_performance
from ensembles.write_to_database import write_to_database


class Bagging:

    def __init__(self, num_base_learner: int, base_learner=PerAlgorithmRegressor(), use_ranking=False, performance_ranking=False, weighting=False, weight_type=None):
        self.logger = logging.getLogger("bagging")
        self.logger.addHandler(logging.StreamHandler())

        # attributes
        self.num_algorithms = 0
        self.base_learners = list()
        self.current_iteration = 0
        self.weights = None

        # parameters
        self.base_learner = base_learner
        self.num_base_learner = num_base_learner
        self.use_ranking = use_ranking
        self.performance_ranking = performance_ranking
        self.weighting = weighting
        self.weight_type = weight_type

    # generate number_of_samples bootstrap samples from the scenario and returns them in a list
    def generate_bootstrap_sample(self, scenario: ASlibScenario, fold: int, number_of_samples: int):
        bootstrap_samples = list()
        out_of_sample_samples = list()
        sample_size = len(scenario.instances)
        random_seed = (fold - 1) * number_of_samples

        for i in range(number_of_samples):
            feature_data_sample = scenario.feature_data.sample(sample_size, replace=True, random_state=random_seed)
            performance_data_sample = scenario.performance_data.sample(sample_size, replace=True, random_state=random_seed)
            runstatus_data_sample = scenario.runstatus_data.sample(sample_size, replace=True, random_state=random_seed)
            feature_runstatus_data_sample = scenario.feature_runstatus_data.sample(sample_size, replace=True, random_state=random_seed)
            if scenario.feature_cost_data is not None:
                feature_cost_data_sample = scenario.feature_cost_data.sample(sample_size, replace=True, random_state=random_seed)
            else:
                feature_cost_data_sample = None

            bootstrap_samples.append((feature_data_sample, performance_data_sample, runstatus_data_sample, feature_runstatus_data_sample, feature_cost_data_sample))
            if self.weighting and self.weight_type == 'oos':
                out_of_sample_samples.append(self.get_out_of_sample(scenario, feature_data_sample.drop_duplicates()))

            random_seed = random_seed + 1

        return bootstrap_samples, out_of_sample_samples

    def get_out_of_sample(self, scenario: ASlibScenario, feature_data_sample):
        out_of_sample = list()
        for instance in scenario.feature_data.index:
            is_in = False
            for index in feature_data_sample.index:
                if instance == index:
                    is_in = True
                    break
            if not is_in:
                out_of_sample.append(instance)

        out_of_sample_feature_data = scenario.feature_data.loc[out_of_sample]
        out_of_sample_performance_data = scenario.performance_data.loc[out_of_sample]
        out_of_sample_runstatus_data = scenario.runstatus_data.loc[out_of_sample]
        out_of_sample_feature_runstatus_data = scenario.feature_runstatus_data.loc[out_of_sample]
        if scenario.feature_cost_data is not None:
            out_of_sample_feature_cost_data = scenario.feature_cost_data.loc[out_of_sample]
        else:
            out_of_sample_feature_cost_data = None

        return (out_of_sample_feature_data, out_of_sample_performance_data, out_of_sample_runstatus_data, out_of_sample_feature_runstatus_data, out_of_sample_feature_cost_data)

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        print("Run fit on " + self.get_name() + " for fold " + str(fold))
        self.num_algorithms = len(scenario.algorithms)

        # create all bootstrap samples
        bootstrap_samples, out_of_sample_samples = self.generate_bootstrap_sample(scenario, fold, self.num_base_learner)

        weights_denorm = list()

        # train each base learner on a different sample
        for index in range(self.num_base_learner):
            self.current_iteration = index + 1
            self.base_learners.append(copy.deepcopy(self.base_learner))
            original_scenario = copy.deepcopy(scenario)
            scenario.feature_data, scenario.performance_data, scenario.runstatus_data, scenario.feature_runstatus_data, scenario.feature_cost_data = bootstrap_samples[index]
            self.base_learners[index].fit(scenario, fold, amount_of_training_instances)
            if self.weighting:
                if self.weight_type == 'oos':
                    scenario.feature_data, scenario.performance_data, scenario.runstatus_data, scenario.feature_runstatus_data, scenario.feature_cost_data = out_of_sample_samples[index]
                elif self.weight_type == 'original_set':
                    scenario = original_scenario
                weights_denorm.append(base_learner_performance(scenario, len(scenario.feature_data), self.base_learners[index]))

            #if self.current_iteration != self.num_base_learner:
            #    write_to_database(scenario, self, fold)

        # Turn around values (lowest (best) gets highest weight) and normalize
        weights_denorm = [max(weights_denorm) / float(i + 1) for i in weights_denorm]
        self.weights = [float(i) / max(weights_denorm) for i in weights_denorm]

    def predict(self, features_of_test_instance, instance_id: int):
        if self.use_ranking:
            if self.weighting:
                return predict_with_ranking(features_of_test_instance, instance_id, self.num_algorithms,
                                            self.base_learners, weights=self.weights, performance_rank=self.performance_ranking)
            else:
                return predict_with_ranking(features_of_test_instance, instance_id, self.num_algorithms,
                                            self.base_learners, performance_rank=self.performance_ranking)

        # only using the prediction of the algorithm
        predictions = np.zeros(self.num_algorithms)
        for i, model in enumerate(self.base_learners):
            # get prediction of base learner and find prediction (lowest value)
            base_prediction = model.predict(features_of_test_instance, instance_id).reshape(self.num_algorithms)
            index_of_minimum = np.where(base_prediction == min(base_prediction))

            # add [1 * weight for base learner] to vote for the algorithm
            if self.weighting:
                predictions[index_of_minimum] = predictions[index_of_minimum] + self.weights[i]
            else:
                predictions[index_of_minimum] = predictions[index_of_minimum] + 1

        return 1 - predictions / sum(predictions)


    def get_name(self):
        name = "bagging_" + str(self.current_iteration) + "_" + self.base_learner.get_name()
        if self.use_ranking:
            name = name + "_ranking"
            if self.performance_ranking:
                name = name + "_averaging"

        if self.weighting:
            name = name + "_weighting"
            if self.weight_type:
                name = name + '_' + self.weight_type

        return name
