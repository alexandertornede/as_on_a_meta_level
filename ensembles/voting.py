import logging

import numpy as np
from survival_approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.isac import ISAC
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from baselines.satzilla11 import SATzilla11
from baselines.sunny import SUNNY
from aslib_scenario.aslib_scenario import ASlibScenario

from ensembles.prediction import predict_with_ranking
from ensembles.validation import base_learner_performance, split_scenario, get_confidence
from ensembles.write_to_file import save_weights
from par_10_metric import Par10Metric
from pre_compute.pickle_loader import load_pickle


class Voting:

    def __init__(self, ranking=False, weighting=False, cross_validation=False, base_learner=None, rank_method='average', pre_computed=False):
        # logger
        self.logger = logging.getLogger("voting")
        self.logger.addHandler(logging.StreamHandler())

        # parameter
        self.ranking = ranking
        self.weighting = weighting
        self.cross_validation = cross_validation
        self.base_learner = base_learner
        self.rank_method = rank_method
        self.pre_computed = pre_computed

        # attributes
        self.trained_models = list()
        self.trained_models_backup = list()
        self.weights = list()
        self.metric = Par10Metric()
        self.num_algorithms = 0
        self.predictions = list()

    def create_base_learner(self):
        # clean up list and init base learners
        self.trained_models = list()

        if 1 in self.base_learner:
            self.trained_models.append(PerAlgorithmRegressor())
        if 2 in self.base_learner:
            self.trained_models.append(SUNNY())
        if 3 in self.base_learner:
            self.trained_models.append(ISAC())
        if 4 in self.base_learner:
            self.trained_models.append(SATzilla11())
        if 5 in self.base_learner:
            self.trained_models.append(SurrogateSurvivalForest(criterion='Expectation'))
        if 6 in self.base_learner:
            self.trained_models.append(SurrogateSurvivalForest(criterion='PAR10'))
        if 7 in self.base_learner:
            self.trained_models.append(MultiClassAlgorithmSelector())


    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        self.num_algorithms = len(scenario.algorithms)
        self.create_base_learner()

        if self.pre_computed:
            for model in self.trained_models:
                self.predictions.append(load_pickle(filename='predictions/' + model.get_name() + '_' + scenario.scenario + '_' + str(fold)))

        if self.cross_validation:
            weights_denorm = np.zeros(len(self.trained_models))
            num_instances = len(scenario.instances)

            # cross validation for the weight
            for sub_fold in range(1, 11):
                test_scenario, training_scenario = split_scenario(scenario, sub_fold, num_instances)
                # train base learner and calculate the weights
                for i, base_learner in enumerate(self.trained_models):
                    base_learner.fit(training_scenario, fold, amount_of_training_instances)
                    weights_denorm[i] = weights_denorm[i] + base_learner_performance(test_scenario, amount_of_training_instances, base_learner)

            # reset trained base learners
            self.create_base_learner()
            # train base learner on the original scenario
            if not self.pre_computed:
                for base_learner in self.trained_models:
                    base_learner.fit(scenario, fold, amount_of_training_instances)

        else:
            weights_denorm = list()

            # train base learner and calculate the weights
            for base_learner in self.trained_models:
                if not self.pre_computed:
                    base_learner.fit(scenario, fold, amount_of_training_instances)
                if self.weighting:
                    if self.pre_computed:
                        predictions = load_pickle(filename='predictions/full_trainingdata_' + base_learner.get_name() + '_' + scenario.scenario + '_' + str(fold))
                        weights_denorm.append(base_learner_performance(scenario, amount_of_training_instances, base_learner, pre_computed_predictions=predictions))
                    else:
                        weights_denorm.append(base_learner_performance(scenario, amount_of_training_instances, base_learner))

        # Turn around values (lowest (best) gets highest weight) and normalize
        weights_denorm = [max(weights_denorm) / float(i + 1) for i in weights_denorm]
        self.weights = [float(i) / max(weights_denorm) for i in weights_denorm]
        if self.weighting:
            save_weights(scenario, fold, self.get_name(), self.weights)

    def predict(self, features_of_test_instance, instance_id: int):

        if self.ranking:
            if self.weighting:
                if self.pre_computed:
                    return predict_with_ranking(features_of_test_instance, instance_id, self.num_algorithms,
                                                self.trained_models, weights=self.weights, pre_computed_predictions=self.predictions)
                else:
                    return predict_with_ranking(features_of_test_instance, instance_id, self.num_algorithms,
                                                self.trained_models, weights=self.weights)
            else:
                if self.pre_computed:
                    return predict_with_ranking(features_of_test_instance, instance_id, self.num_algorithms,
                                                self.trained_models, weights=None, rank_method=self.rank_method, pre_computed_predictions=self.predictions)
                else:
                    return predict_with_ranking(features_of_test_instance, instance_id, self.num_algorithms,
                                                self.trained_models, weights=None, rank_method=self.rank_method)

        # only using the prediction of the algorithm
        predictions = np.zeros(self.num_algorithms)

        for learner_index, model in enumerate(self.trained_models):

            # get prediction of base learner and find prediction (lowest value)
            if not self.pre_computed:
                base_prediction = model.predict(features_of_test_instance, instance_id).reshape(self.num_algorithms)
            else:
                base_prediction = self.predictions[learner_index][str(features_of_test_instance)]

            index_of_minimum = np.argmin(base_prediction)

            # add [1 * weight for base learner] to vote for the algorithm
            if self.weighting:
                predictions[index_of_minimum] = predictions[index_of_minimum] + self.weights[learner_index]
            else:
                predictions[index_of_minimum] = predictions[index_of_minimum] + 1

        return 1 - predictions / sum(predictions)

    def get_name(self):
        name = "voting"
        if self.ranking:
            name = name + "_ranking"
        if self.rank_method != 'average':
            name = name + "_" + self.rank_method
        if self.weighting:
            name = name + "_weighting"
        if self.cross_validation:
            name = name + "_cross"
        if self.base_learner:
            name = name + "_" + str(self.base_learner).replace('[', '').replace(']', '').replace(', ', '_')
        return name
