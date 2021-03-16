from aslib_scenario.aslib_scenario import ASlibScenario
import numpy as np

from evaluation import publish_results_to_database
from number_unsolved_instances import NumberUnsolvedInstances
from par_10_metric import Par10Metric
import configparser


def _evaluate_train_test_split_mod(scenario: ASlibScenario, approach, metrics, fold: int):
    test_scenario, train_scenario = scenario.get_split(indx=fold)

    approach_metric_values = np.zeros(len(metrics))

    num_counted_test_values = 0

    feature_data = test_scenario.feature_data.to_numpy()
    performance_data = test_scenario.performance_data.to_numpy()
    feature_cost_data = test_scenario.feature_cost_data.to_numpy() if test_scenario.feature_cost_data is not None else None

    for instance_id in range(0, len(test_scenario.instances)):
        X_test = feature_data[instance_id]
        y_test = performance_data[instance_id]

        accumulated_feature_time = 0
        if test_scenario.feature_cost_data is not None and approach.get_name() != 'sbs' and approach.get_name() != 'oracle':
            feature_time = feature_cost_data[instance_id]
            accumulated_feature_time = np.sum(feature_time)

        contains_non_censored_value = False
        for y_element in y_test:
            if y_element < test_scenario.algorithm_cutoff_time:
                contains_non_censored_value = True
        if contains_non_censored_value:
            num_counted_test_values += 1
            predicted_scores = approach.predict(X_test, instance_id)
            for i, metric in enumerate(metrics):
                runtime = metric.evaluate(y_test, predicted_scores, accumulated_feature_time, scenario.algorithm_cutoff_time)
                approach_metric_values[i] = (approach_metric_values[i] + runtime)

    approach_metric_values = np.true_divide(approach_metric_values, num_counted_test_values)

    print('PAR10: {0:.10f}'.format(approach_metric_values[0]))

    return approach_metric_values

def write_to_database(scenario: ASlibScenario, approach, fold: int):
    metrics = list()
    metrics.append(Par10Metric())
    metrics.append(NumberUnsolvedInstances(False))
    metrics.append(NumberUnsolvedInstances(True))
    scenario_name = scenario.scenario
    scenario = ASlibScenario()
    if scenario_name == 'GLUHACK-18':
        scenario_name = 'GLUHACK-2018'
    scenario.read_scenario('data/aslib_data-master/' + scenario_name)
    metric_results = _evaluate_train_test_split_mod(scenario, approach, metrics, fold)

    db_config = load_configuration()
    for i, result in enumerate(metric_results):
        publish_results_to_database(db_config, scenario.scenario, fold, approach.get_name(), metrics[i].get_name(), result)

def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('conf/experiment_configuration.cfg'))
    return config