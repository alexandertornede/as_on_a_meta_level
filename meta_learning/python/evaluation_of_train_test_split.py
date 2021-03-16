import copy
import logging
import numpy as np
import os
from aslib_scenario.aslib_scenario import ASlibScenario
from simple_runtime_metric import RuntimeMetric

logger = logging.getLogger("evaluate_train_test_split")
logger.addHandler(logging.StreamHandler())


def evaluate_train_test_split(scenario: ASlibScenario, approach, metrics, fold: int, amount_of_training_instances: int, train_status:str):
    test_scenario, train_scenario = scenario.get_split(indx=fold)

    if train_status != 'all':
        train_scenario = copy.deepcopy(train_scenario)
        threshold = train_scenario.algorithm_cutoff_time
        if train_status == 'clip_censored':
            train_scenario.performance_data = train_scenario.performance_data.clip(upper=threshold)

        elif train_status == 'ignore_censored':
            train_scenario.performance_data = train_scenario.performance_data.replace(10*threshold, np.nan)

    if approach.get_name() == 'oracle' or approach.get_name() == 'virtual_sbs_with_feature_costs':
        approach.fit(test_scenario, fold, amount_of_training_instances)
    else:
        approach.fit(train_scenario, fold, amount_of_training_instances)

    approach_metric_values = np.zeros(len(metrics))

    num_counted_test_values = 0

    feature_data = test_scenario.feature_data.to_numpy()
    performance_data = test_scenario.performance_data.to_numpy()
    feature_cost_data = test_scenario.feature_cost_data.to_numpy() if test_scenario.feature_cost_data is not None else None

    instancewise_result_strings = list()
    simple_runtime_metric = RuntimeMetric()

    for instance_id in range(0, len(test_scenario.instances)):

        X_test = feature_data[instance_id]
        y_test = performance_data[instance_id]

        # compute feature time
        accumulated_feature_time = 0
        if test_scenario.feature_cost_data is not None and approach.get_name() != 'sbs' and approach.get_name() != 'oracle':
            feature_time = feature_cost_data[instance_id]
            accumulated_feature_time = np.sum(feature_time)


        #compute the values of the different metrics
        predicted_scores = approach.predict(X_test, instance_id)
        num_counted_test_values += 1
        for i, metric in enumerate(metrics):
            runtime = metric.evaluate(y_test, predicted_scores, accumulated_feature_time, scenario.algorithm_cutoff_time)
            approach_metric_values[i] = (approach_metric_values[i] + runtime)

        # store runtimes on a per instance basis in ASLib format
        runtime = simple_runtime_metric.evaluate(y_test, predicted_scores, accumulated_feature_time, scenario.algorithm_cutoff_time)
        run_status_to_print = "ok" if runtime < scenario.algorithm_cutoff_time else "timeout"
        line_to_store = test_scenario.instances[instance_id] + ",1," + approach.get_name() + "," + str(runtime) + "," + run_status_to_print
        instancewise_result_strings.append(line_to_store)

    write_instance_wise_results_to_file(instancewise_result_strings, scenario.scenario)


    approach_metric_values = np.true_divide(approach_metric_values, num_counted_test_values)

    for i, metric in enumerate(metrics):
        print(metrics[i].get_name() + ': {0:.10f}'.format(approach_metric_values[i]))

    return approach_metric_values


def write_instance_wise_results_to_file(instancewise_result_strings: list, scenario_name: str):
    if not os.path.exists('output'):
        os.makedirs('output')
    complete_instancewise_result_string = '\n'.join(instancewise_result_strings)
    f = open("output/" + scenario_name + ".arff", "a")
    f.write(complete_instancewise_result_string + "\n")
    f.close()