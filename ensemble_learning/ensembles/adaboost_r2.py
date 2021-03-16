import copy
import logging
import sys
import math
import numpy as np

from survival_approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from aslib_scenario.aslib_scenario import ASlibScenario
from ensembles.write_to_database import write_to_database


class AdaboostR2:

    def __init__(self, algorithm_name, max_iterations=20, loss_function='linear', different_error_type=False):
        # setup
        self.logger = logging.getLogger("boosting")
        self.logger.addHandler(logging.StreamHandler())

        # parameters
        self.algorithm_name = algorithm_name
        self.max_iterations = max_iterations
        self.loss_function = loss_function
        self.different_error_type = different_error_type

        # attributes
        self.current_iteration = 0
        self.base_learners = list()
        self.beta = list()
        self.data_weights = list()

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        # setup values
        actual_num_training_instances = amount_of_training_instances if amount_of_training_instances != -1 else len(scenario.instances)
        self.data_weights = np.ones(actual_num_training_instances)

        # boosting iterations (stop when avg_loss >= 0.5 or iteration = max_iterations)
        for iteration in range(self.max_iterations):
            self.current_iteration = self.current_iteration + 1

            # choose base learner algorithm
            if self.algorithm_name == 'per_algorithm_regressor':
                self.base_learners.append(PerAlgorithmRegressor())
            elif self.algorithm_name == 'par10':
                self.base_learners.append(SurrogateSurvivalForest(criterion='PAR10'))
            else:
                sys.exit('Wrong base learner for boosting')

            # get weighted scenario and train new base learner
            new_scenario = self.generate_weighted_sample(scenario, fold, actual_num_training_instances)
            self.base_learners[iteration].fit(new_scenario, fold, amount_of_training_instances)

            # calculate weights for next iteration
            if not self.update_weights(scenario, self.base_learners[iteration], actual_num_training_instances):
                break

            # use write_to_database to write each iteration to database
            #if self.current_iteration < self.max_iterations:
            #    write_to_database(scenario, self, fold)

    def predict(self, features_of_test_instance, instance_id: int):
        # get the predictions and confidence values from each base learner
        y_predictions = list()
        for base_learner, beta in zip(self.base_learners, self.beta):
            y_predictions.append((np.amin(base_learner.predict(features_of_test_instance, instance_id)), beta, base_learner))
        # sort base learners based on prediction value
        y_predictions.sort(key=lambda x: x[0])

        # set lower_bound for prediction
        lower_bound = 0.0
        for beta in y_predictions:
            lower_bound = lower_bound + math.log(1/beta[1])
        lower_bound = lower_bound * 0.5

        # sum beta while beta_sum < lower_bound -> this is the weighted median
        beta_sum = 0.0
        for t, beta in enumerate(y_predictions):
            beta_sum = beta_sum + math.log(1 / beta[1])
            if beta_sum >= lower_bound:
                prediction = y_predictions[t][2].predict(features_of_test_instance, instance_id)
                return prediction

    def update_weights(self, scenario: ASlibScenario, base_learner, amount_of_training_instances: int):
        # get data from original scenario
        feature_data = scenario.feature_data.to_numpy()
        performance_data = scenario.performance_data.to_numpy()
        feature_cost_data = scenario.feature_cost_data.to_numpy() if scenario.feature_cost_data is not None else None

        temp_loss = list()

        for instance_id in range(amount_of_training_instances):
            # all instances are for testing
            x_test = feature_data[instance_id]
            y_test = performance_data[instance_id]

            accumulated_feature_time = 0
            if scenario.feature_cost_data is not None:
                feature_time = feature_cost_data[instance_id]
                accumulated_feature_time = np.sum(feature_time)

            # contains_non_censored_value = False
            # for y_element in y_test:
            #    if y_element < test_scenario.algorithm_cutoff_time:
            #        contains_non_censored_value = True
            # if contains_non_censored_value:
            #    num_counted_test_values += 1

            # calculate loss function for each instance
            predictions = base_learner.predict(x_test, instance_id)
            if self.different_error_type:
                temp_loss.append(abs(np.amin(predictions) - np.amin(y_test)))
            else:
                y_min = np.argmin(y_test)
                temp_loss.append(abs(float(predictions[y_min] - y_test[y_min])))

        # calculate loss function for base learner
        if self.loss_function == 'linear':
            loss = temp_loss / np.amax(temp_loss)
        elif self.loss_function == 'square':
            loss = [i ** 2 for i in temp_loss] / (np.amax(temp_loss) ** 2)
        elif self.loss_function == 'exponential':
            loss = np.zeros(len(temp_loss))
            for i in range(len(temp_loss)):
                loss[i] = 1 - math.exp(-temp_loss[i] / np.amax(temp_loss))
        else:
            sys.exit("Unknown loss function")

        # calculate average loss
        avg_loss = sum(loss * (self.data_weights / sum(self.data_weights)))

        # calculate confidence -> lower = better
        beta = avg_loss / (1 - avg_loss)
        self.beta.append(beta)

        # update weights
        self.data_weights = self.data_weights * beta**(1-loss)

        # check if average loss function is below 0.5
        if avg_loss >= 0.5:
            return False
        return True

    def generate_weighted_sample(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        # copy original scenario
        new_scenario = copy.deepcopy(scenario)

        # create weighted sample
        new_scenario.feature_data = new_scenario.feature_data.sample(amount_of_training_instances, replace=True, weights=self.data_weights, random_state=fold)
        new_scenario.performance_data = new_scenario.performance_data.sample(amount_of_training_instances, replace=True, weights=self.data_weights, random_state=fold)
        if scenario.feature_cost_data is not None:
            new_scenario.feature_cost_data = new_scenario.feature_cost_data.sample(amount_of_training_instances, replace=True, weights=self.data_weights, random_state=fold)

        return new_scenario

    def get_name(self):
        name = "adaboostR2_" + self.algorithm_name + "_" + self.loss_function + '_' + str(self.current_iteration)
        if self.different_error_type:
            # se = smallest error
            name = name + '_se'
        return name
