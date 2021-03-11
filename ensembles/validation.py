import copy

from aslib_scenario.aslib_scenario import ASlibScenario
import numpy as np
import math

from par_10_metric import Par10Metric


def base_learner_performance(scenario: ASlibScenario, amount_of_training_instances: int, base_learner, pre_computed_predictions=None):
    # extract data from scenario
    feature_data = scenario.feature_data.to_numpy()
    performance_data = scenario.performance_data.to_numpy()
    feature_cost_data = scenario.feature_cost_data.to_numpy() if scenario.feature_cost_data is not None else None
    metric = Par10Metric()
    num_iterations = len(scenario.instances) if amount_of_training_instances == -1 else amount_of_training_instances

    # performance_measure hold the PAR10 score for every instance
    performance_measure = 0
    for instance_id in range(num_iterations):
        x_test = feature_data[instance_id]
        y_test = performance_data[instance_id]

        accumulated_feature_time = 0
        if scenario.feature_cost_data is not None:
            feature_time = feature_cost_data[instance_id]
            accumulated_feature_time = np.sum(feature_time)

        if pre_computed_predictions is not None:
            predicted_scores = pre_computed_predictions[instance_id]
        else:
            predicted_scores = base_learner.predict(x_test, instance_id)
        performance_measure = performance_measure + metric.evaluate(y_test, predicted_scores,
                                                                    accumulated_feature_time,
                                                                    scenario.algorithm_cutoff_time)
    return performance_measure / num_iterations

def get_confidence(scenario: ASlibScenario, amount_of_training_instances: int, base_learner):
        feature_data = scenario.feature_data.to_numpy()
        performance_data = scenario.performance_data.to_numpy()
        feature_cost_data = scenario.feature_cost_data.to_numpy() if scenario.feature_cost_data is not None else None

        err = 0

        for instance_id in range(amount_of_training_instances):
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

            predictions = base_learner.predict(x_test, instance_id)
            y_algorithm = np.argmin(y_test)
            predicted_algorithm = np.argmin(predictions)
            if y_algorithm != predicted_algorithm:
                err = err + 1

        err = err / amount_of_training_instances

        if err < 0.001:
            err = 0.001
    
        confidence = math.log((1 - err) / err) + math.log(len(scenario.algorithms) - 1)
        return confidence


def split_scenario(scenario: ASlibScenario, sub_fold: int, num_instances: int):
    fold_len = int(num_instances / 10)
    instances = scenario.instances
    if sub_fold < 10:
        test_insts = instances[(sub_fold - 1) * fold_len:sub_fold * fold_len]
        training_insts = instances[:(sub_fold - 1) * fold_len]
        training_insts = np.append(training_insts, instances[sub_fold * fold_len:])
    else:
        test_insts = instances[(sub_fold - 1) * fold_len:]
        training_insts = instances[:(sub_fold - 1) * fold_len]

    test = copy.copy(scenario)
    training = copy.copy(scenario)

    # feature_data
    test.feature_data = test.feature_data.drop(training_insts).sort_index()
    training.feature_data = training.feature_data.drop(test_insts).sort_index()

    # performance_data
    test.performance_data = test.performance_data.drop(training_insts).sort_index()
    training.performance_data = training.performance_data.drop(test_insts).sort_index()

    # runstatus_data
    test.runstatus_data = test.runstatus_data.drop(training_insts).sort_index()
    training.runstatus_data = training.runstatus_data.drop(test_insts).sort_index()

    # feature_runstatus_data
    test.feature_runstatus_data = test.feature_runstatus_data.drop(training_insts).sort_index()
    training.feature_runstatus_data = training.feature_runstatus_data.drop(test_insts).sort_index()

    # feature_cost_data
    if scenario.feature_cost_data is not None:
        test.feature_cost_data = test.feature_cost_data.drop(training_insts).sort_index()
        training.feature_cost_data = training.feature_cost_data.drop(test_insts).sort_index()

    # ground_truth_data
    if scenario.ground_truth_data is not None:
        test.ground_truth_data = test.ground_truth_data.drop(training_insts).sort_index()
        training.ground_truth_data = training.ground_truth_data.drop(test_insts).sort_index()

    test.cv_data = None
    training.cv_data = None

    test.instances = test_insts
    training.instances = training_insts

    scenario.used_feature_groups = None

    return test, training
