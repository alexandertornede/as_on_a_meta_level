from aslib_scenario.aslib_scenario import ASlibScenario
import numpy as np
import logging
import database_utils
from evaluation_of_train_test_split import evaluate_train_test_split
from hyperparameter_optimizer import HyperParameterOptimizer

logger = logging.getLogger("evaluation")
logger.addHandler(logging.StreamHandler())


def publish_results_to_database(db_config, scenario_name: str, fold: int, approach: str, metric_name:str, result: float):
    db_handle, table_name = database_utils.initialize_mysql_db_and_table_name_from_config(db_config)

    db_cursor = db_handle.cursor()
    sql_statement = "INSERT INTO " + table_name + " (scenario_name, fold, approach, metric, result) VALUES (%s, %s, %s, %s, %s)"
    values = (scenario_name, fold, approach, metric_name, str(result))
    db_cursor.execute(sql_statement, values)
    db_handle.commit()

    db_cursor.close()
    db_handle.close()


def evaluate(scenario: ASlibScenario, approach, metrics, amount_of_training_instances: int, fold: int, db_config, tune_hyperparameters: bool):
    np.random.seed(fold)

    logger.info("-----------------------------")
    logger.info("Evaluating \"" + approach.get_name() + "\" fold " + str(fold) + " training on " + str(
        amount_of_training_instances) + " scenario instances on scenario " + str(scenario.scenario))

    if tune_hyperparameters:
        optimizer = HyperParameterOptimizer()
        parametrization = optimizer.optimize(scenario, approach)
        approach.set_parameters(parametrization)

    train_status = db_config["EXPERIMENTS"]["train_status"]
    metric_results = evaluate_train_test_split(scenario, approach, metrics, fold, amount_of_training_instances, train_status)

    for i, result in enumerate(metric_results):
        publish_results_to_database(db_config, scenario.scenario, fold, approach.get_name(), metrics[i].get_name(), result)


def print_stats_of_scenario(scenario: ASlibScenario):
    logger.info("scenario: " + str(scenario.scenario))
    logger.info("#instances: " + str(len(scenario.instances)))
    logger.info("#features: " + str(len(scenario.feature_data.columns)))
    logger.info("#algorithms: " + str(len(scenario.algorithms)))
    logger.info("cutoff-time: " + str(scenario.algorithm_cutoff_time))


def evaluate_scenario(scenario_name: str, path_to_scenario_folder:str, approach, metrics, amount_of_training_scenario_instances: int, fold: int, db_config, tune_hyperparameters: bool):
    scenario = ASlibScenario()
    scenario.read_scenario(path_to_scenario_folder + scenario_name)
    print_stats_of_scenario(scenario)
    evaluate(scenario, approach, metrics, amount_of_training_scenario_instances, fold, db_config, tune_hyperparameters)
    return scenario_name




