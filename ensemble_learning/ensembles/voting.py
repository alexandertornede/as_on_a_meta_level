import itertools
import logging
import sys

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

    def __init__(self, ranking=False, weighting=False, cross_validation=False, base_learner=None, rank_method='average', pre_computed=False, optimze_base_learner=False):
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
        self.left_out_learners = []
        self.optimze_base_learner = optimze_base_learner

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
        if self.optimze_base_learner:
            self.set_left_out_base_learner(scenario, fold)

        if self.pre_computed:
            self.predictions = []
            for model in self.trained_models:
                self.predictions.append(load_pickle(filename='predictions/' + model.get_name() + '_' + scenario.scenario + '_' + str(fold)))

    def set_left_out_base_learner(self, scenario: ASlibScenario, fold: int):
        self.predictions = []
        for base_learner in self.trained_models:
            if self.pre_computed:
                self.predictions.append(load_pickle(filename='predictions/full_trainingdata_' + base_learner.get_name() + '_' + scenario.scenario + '_' + str(fold)))
            else:
                # TODO: Implement wto use without pre_computed predictions
                sys.exit("Not implemented")
        t = range(len(self.trained_models))
        self.left_out_learners = []

        best_combination = self.left_out_learners
        best_performance = self.get_par10(scenario, fold)
        for x in range(1, len(t) - 1):
            c = list(itertools.combinations(t, x))
            unq = set(c)
            for u in unq:
                self.left_out_learners = list(u)

                cur_performance = self.get_par10(scenario, fold)
                if cur_performance < best_performance:
                    best_performance = cur_performance
                    best_combination = self.left_out_learners

        self.left_out_learners = best_combination

    def get_par10(self, scenario: ASlibScenario, fold: int):
        metrics = list()
        metrics.append(Par10Metric())

        test_scenario, train_scenario = scenario.get_split(indx=fold)

        approach_metric_values = np.zeros(len(metrics))

        num_counted_test_values = 0

        feature_data = train_scenario.feature_data.to_numpy()
        performance_data = train_scenario.performance_data.to_numpy()
        feature_cost_data = train_scenario.feature_cost_data.to_numpy() if train_scenario.feature_cost_data is not None else None

        for instance_id in range(0, len(train_scenario.instances)):
            X_test = feature_data[instance_id]
            y_test = performance_data[instance_id]

            accumulated_feature_time = 0
            if train_scenario.feature_cost_data is not None and self.get_name() != 'sbs' and self.get_name() != 'oracle':
                feature_time = feature_cost_data[instance_id]
                accumulated_feature_time = np.sum(feature_time)

            contains_non_censored_value = False
            for y_element in y_test:
                if y_element < train_scenario.algorithm_cutoff_time:
                    contains_non_censored_value = True
            if contains_non_censored_value:
                num_counted_test_values += 1
                predicted_scores = self.predict(X_test, instance_id, opt=True)
                for i, metric in enumerate(metrics):
                    runtime = metric.evaluate(y_test, predicted_scores, accumulated_feature_time,
                                              scenario.algorithm_cutoff_time)
                    approach_metric_values[i] = (approach_metric_values[i] + runtime)

        approach_metric_values = np.true_divide(approach_metric_values, num_counted_test_values)

        return approach_metric_values

    def predict(self, features_of_test_instance, instance_id: int, opt=False):

        models = []
        model_index = []
        for i, model in enumerate(self.trained_models):
            if i not in self.left_out_learners:
                models.append(model)
                model_index.append(i)


        if self.ranking:
            if self.weighting:
                if self.pre_computed:
                    return predict_with_ranking(features_of_test_instance, instance_id, self.num_algorithms,
                                                models, model_index=model_index, weights=self.weights, pre_computed_predictions=self.predictions, opt=opt)
                else:
                    return predict_with_ranking(features_of_test_instance, instance_id, self.num_algorithms,
                                                models, model_index=model_index, weights=self.weights, opt=opt)
            else:
                if self.pre_computed:
                    return predict_with_ranking(features_of_test_instance, instance_id, self.num_algorithms,
                                                models, model_index=model_index, weights=None, rank_method=self.rank_method, pre_computed_predictions=self.predictions, opt=opt)
                else:
                    return predict_with_ranking(features_of_test_instance, instance_id, self.num_algorithms,
                                                models, model_index=model_index, weights=None, rank_method=self.rank_method, opt=opt)

        # only using the prediction of the algorithm
        predictions = np.zeros(self.num_algorithms)

        for learner_index, model in zip(model_index, models):

            # get prediction of base learner and find prediction (lowest value)
            if not self.pre_computed:
                base_prediction = model.predict(features_of_test_instance, instance_id).reshape(self.num_algorithms)
            else:
                if opt:
                    base_prediction = self.predictions[learner_index][instance_id]
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
        if self.optimze_base_learner:
            learner = list(range(1, len(self.trained_models) + 1))
            for i in self.left_out_learners:
                learner.remove(i + 1)
            name = name + 'left_out_base_learner' + str(learner).replace('[', '').replace(']', '').replace(', ', '_')
        else:
            if self.base_learner:
                name = name + "_" + str(self.base_learner).replace('[', '').replace(']', '').replace(', ', '_')
        return name
