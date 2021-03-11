import copy
import logging
import numpy as np
from aslib_scenario.aslib_scenario import ASlibScenario

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

    if approach.get_name() == 'oracle':
        approach.fit(test_scenario, fold, amount_of_training_instances)
    else:
        approach.fit(train_scenario, fold, amount_of_training_instances)

    approach_metric_values = np.zeros(len(metrics))

    num_counted_test_values = 0

    feature_data = test_scenario.feature_data.to_numpy()
    performance_data = test_scenario.performance_data.to_numpy()
    feature_cost_data = test_scenario.feature_cost_data.to_numpy() if test_scenario.feature_cost_data is not None else None

    for instance_id in range(0, len(test_scenario.instances)):

        #logger.debug("Test instance " + str(instance_id) + "/" + str(len(test_scenario.instances)))

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