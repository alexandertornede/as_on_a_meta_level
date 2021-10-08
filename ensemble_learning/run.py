import os

os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

import logging
import configparser
import multiprocessing as mp

import database_utils
from ensembles.bagging import Bagging
from ensembles.samme import SAMME
from ensembles.stacking import Stacking
from pre_compute.run_pre_computed_base_learner import RunPreComputedBaseLearner
from pre_compute.create_base_learner_predictions import CreateBaseLearnerPrediction
from ensembles.voting import Voting
from evaluation import evaluate_scenario
from survival_approaches.single_best_solver import SingleBestSolver
from survival_approaches.oracle import Oracle
from survival_approaches.survival_forests.surrogate import SurrogateSurvivalForest
from survival_approaches.survival_forests.auto_surrogate import SurrogateAutoSurvivalForest
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.sunny import SUNNY
from baselines.snnap import SNNAP
from baselines.isac import ISAC
from baselines.satzilla11 import SATzilla11
from baselines.satzilla07 import SATzilla07
from sklearn.linear_model import Ridge
from par_10_metric import Par10Metric
from number_unsolved_instances import NumberUnsolvedInstances

logger = logging.getLogger("run")
logger.addHandler(logging.StreamHandler())


def initialize_logging():
    logging.basicConfig(filename='logs/log_file.log', filemode='w',
                        format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S %p', level=logging.DEBUG)


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('conf/experiment_configuration.cfg'))
    return config


def print_config(config: configparser.ConfigParser):
    for section in config.sections():
        logger.info(str(section) + ": " + str(dict(config[section])))


def log_result(result):
    logger.info("Finished experiements for scenario: " + result)


def generate_combinations(base_lerners, combinations):
    if base_lerners == [] or len(base_lerners) < 2:
        return []
    else:
        combinations.append(base_lerners)
        for lerner in range(len(base_lerners)):
            base_lerner_copy = list(base_lerners)
            del base_lerner_copy[lerner]
            generate_combinations(base_lerner_copy, combinations)
        return combinations


def get_combinations(base_lerners):
    combinations = list()
    for combination in set(tuple(i) for i in generate_combinations(base_lerners, [])):
        combinations.append(list(combination))
    return combinations


def create_approach(approach_names):
    approaches = list()
    for approach_name in approach_names:

        # SBS and VBS
        if approach_name == 'sbs':
            approaches.append(SingleBestSolver())
        if approach_name == 'oracle':
            approaches.append(Oracle())

        # baselines
        if approach_name == 'ExpectationSurvivalForest':
            approaches.append(SurrogateSurvivalForest(criterion='Expectation'))
        if approach_name == 'PolynomialSurvivalForest':
            approaches.append(SurrogateSurvivalForest(criterion='Polynomial'))
        if approach_name == 'GridSearchSurvivalForest':
            approaches.append(SurrogateSurvivalForest(criterion='GridSearch'))
        if approach_name == 'ExponentialSurvivalForest':
            approaches.append(SurrogateSurvivalForest(criterion='Exponential'))
        if approach_name == 'SurrogateAutoSurvivalForest':
            approaches.append(SurrogateAutoSurvivalForest())
        if approach_name == 'PAR10SurvivalForest':
            approaches.append(SurrogateSurvivalForest(criterion='PAR10'))
        if approach_name == 'per_algorithm_regressor':
            approaches.append(PerAlgorithmRegressor())
        if approach_name == 'imputed_per_algorithm_rf_regressor':
            approaches.append(PerAlgorithmRegressor(impute_censored=True))
        if approach_name == 'imputed_per_algorithm_ridge_regressor':
            approaches.append(PerAlgorithmRegressor(
                scikit_regressor=Ridge(alpha=1.0), impute_censored=True))
        if approach_name == 'multiclass_algorithm_selector':
            approaches.append(MultiClassAlgorithmSelector())
        if approach_name == 'sunny':
            approaches.append(SUNNY())
        if approach_name == 'snnap':
            approaches.append(SNNAP())
        if approach_name == 'satzilla-11':
            approaches.append(SATzilla11())
        if approach_name == 'satzilla-07':
            approaches.append(SATzilla07())
        if approach_name == 'isac':
            approaches.append(ISAC())

        if approach_name == 'base_learner':
            approaches.append(RunPreComputedBaseLearner('per_algorithm_RandomForestRegressor_regressor'))
            approaches.append(RunPreComputedBaseLearner('sunny'))
            approaches.append(RunPreComputedBaseLearner('isac'))
            approaches.append(RunPreComputedBaseLearner('satzilla-11'))
            approaches.append(RunPreComputedBaseLearner('Expectation_algorithm_survival_forest'))
            approaches.append(RunPreComputedBaseLearner('PAR10_algorithm_survival_forest'))
            approaches.append(RunPreComputedBaseLearner('multiclass_algorithm_selector'))

        # voting
        if approach_name == 'voting':
            for combination in get_combinations([1, 2, 3, 4, 5, 6, 7]):
                approaches.append(Voting(base_learner=combination, pre_computed=True))
        if approach_name == 'voting_borda':
            for combination in get_combinations([1, 2, 3, 4, 5, 6, 7]):
                approaches.append(Voting(base_learner=combination, ranking=True, pre_computed=True))
        if approach_name == 'voting_weighting':
            for combination in get_combinations([1, 2, 3, 4, 5, 6, 7]):
                approaches.append(Voting(base_learner=combination, pre_computed=True, weighting=True))
        if approach_name == 'voting_optimize':
            approaches.append(Voting(base_learner=[1, 2, 3, 4, 5, 6, 7], pre_computed=True, optimze_base_learner=True))
            approaches.append(
                Voting(base_learner=[1, 2, 3, 4, 5, 6, 7], ranking=True, pre_computed=True, optimze_base_learner=True))
            approaches.append(Voting(base_learner=[1, 2, 3, 4, 5, 6, 7], weighting=True, pre_computed=True,
                                     optimze_base_learner=True))

        # bagging
        if approach_name == 'bagging-base_learner':
            approaches.append(Bagging(num_base_learner=10, base_learner=PerAlgorithmRegressor()))
            approaches.append(Bagging(num_base_learner=10, base_learner=SUNNY()))
            approaches.append(Bagging(num_base_learner=10, base_learner=ISAC()))
            approaches.append(Bagging(num_base_learner=10, base_learner=SATzilla11()))
            approaches.append(Bagging(num_base_learner=10, base_learner=MultiClassAlgorithmSelector()))

        if approach_name == 'bagging_weighting-base_learner':
            approaches.append(Bagging(num_base_learner=10, base_learner=PerAlgorithmRegressor(), weighting=True))
            approaches.append(Bagging(num_base_learner=10, base_learner=SUNNY(), weighting=True))
            approaches.append(Bagging(num_base_learner=10, base_learner=ISAC(), weighting=True))
            approaches.append(Bagging(num_base_learner=10, base_learner=SATzilla11(), weighting=True))
            approaches.append(Bagging(num_base_learner=10, base_learner=MultiClassAlgorithmSelector(), weighting=True))

        if approach_name == 'bagging_borda-base_learner':
            approaches.append(Bagging(num_base_learner=10, base_learner=PerAlgorithmRegressor(), use_ranking=True))
            approaches.append(Bagging(num_base_learner=10, base_learner=SUNNY(), use_ranking=True))
            approaches.append(Bagging(num_base_learner=10, base_learner=ISAC(), use_ranking=True))
            approaches.append(Bagging(num_base_learner=10, base_learner=SATzilla11(), use_ranking=True))
            approaches.append(
                Bagging(num_base_learner=10, base_learner=MultiClassAlgorithmSelector(), use_ranking=True))

        # boosting
        if approach_name == 'samme':
            approaches.append(SAMME('per_algorithm_regressor', num_iterations=20))
            approaches.append(SAMME('multiclass_algorithm_selector', num_iterations=20))
            approaches.append(SAMME('sunny', num_iterations=20))
            approaches.append(SAMME('isac', num_iterations=20))

        # stacking
        if approach_name == 'stacking_meta_learner':
            base_learner = [1, 2, 3, 4, 5, 6, 7]
            approaches.append(
                Stacking(base_learner=base_learner, meta_learner_type='per_algorithm_regressor', pre_computed=True))
            approaches.append(Stacking(base_learner=base_learner, meta_learner_type='SUNNY', pre_computed=True))
            approaches.append(Stacking(base_learner=base_learner, meta_learner_type='ISAC', pre_computed=True))
            approaches.append(Stacking(base_learner=base_learner, meta_learner_type='SATzilla-11', pre_computed=True))
            approaches.append(Stacking(base_learner=base_learner, meta_learner_type='PAR10', pre_computed=True))
            approaches.append(Stacking(base_learner=base_learner, meta_learner_type='multiclass', pre_computed=True))
        if approach_name == 'stacking_feature_selection':
            base_learner = [1, 2, 3, 4, 5, 6, 7]
            approaches.append(
                Stacking(base_learner=base_learner, meta_learner_type='per_algorithm_regressor', pre_computed=True,
                         feature_selection='variance_threshold'))
            approaches.append(Stacking(base_learner=base_learner, meta_learner_type='SUNNY', pre_computed=True,
                                       feature_selection='variance_threshold'))
            approaches.append(Stacking(base_learner=base_learner, meta_learner_type='ISAC', pre_computed=True,
                                       feature_selection='variance_threshold'))
            approaches.append(Stacking(base_learner=base_learner, meta_learner_type='SATzilla-11', pre_computed=True,
                                       feature_selection='variance_threshold'))
            approaches.append(Stacking(base_learner=base_learner, meta_learner_type='PAR10', pre_computed=True,
                                       feature_selection='variance_threshold'))
            approaches.append(Stacking(base_learner=base_learner, meta_learner_type='multiclass', pre_computed=True,
                                       feature_selection='variance_threshold'))

        # precompute baseline predictions
        if approach_name == 'create_base_learner_prediction':
            approaches.append(
                CreateBaseLearnerPrediction(algorithm='per_algorithm_regressor', for_cross_validation=False,
                                            predict_full_training_set=True))
            approaches.append(
                CreateBaseLearnerPrediction(algorithm='per_algorithm_regressor', for_cross_validation=False))

            approaches.append(CreateBaseLearnerPrediction(algorithm='sunny', for_cross_validation=False,
                                                          predict_full_training_set=True))
            approaches.append(CreateBaseLearnerPrediction(algorithm='sunny', for_cross_validation=False))

            approaches.append(CreateBaseLearnerPrediction(algorithm='isac', for_cross_validation=False,
                                                          predict_full_training_set=True))
            approaches.append(CreateBaseLearnerPrediction(algorithm='isac', for_cross_validation=False))

            approaches.append(CreateBaseLearnerPrediction(algorithm='satzilla', for_cross_validation=False,
                                                          predict_full_training_set=True))
            approaches.append(CreateBaseLearnerPrediction(algorithm='satzilla', for_cross_validation=False))

            approaches.append(CreateBaseLearnerPrediction(algorithm='expectation', for_cross_validation=False,
                                                          predict_full_training_set=True))
            approaches.append(CreateBaseLearnerPrediction(algorithm='expectation', for_cross_validation=False))

            approaches.append(CreateBaseLearnerPrediction(algorithm='par10', for_cross_validation=False,
                                                          predict_full_training_set=True))
            approaches.append(CreateBaseLearnerPrediction(algorithm='par10', for_cross_validation=False))

            approaches.append(CreateBaseLearnerPrediction(algorithm='multiclass', for_cross_validation=False,
                                                          predict_full_training_set=True))
            approaches.append(CreateBaseLearnerPrediction(algorithm='multiclass', for_cross_validation=False))

    return approaches


#######################
#         MAIN        #
#######################

initialize_logging()
config = load_configuration()
logger.info("Running experiments with config:")
print_config(config)

db_handle, table_name = database_utils.initialize_mysql_db_and_table_name_from_config(
    config)
database_utils.create_table_if_not_exists(db_handle, table_name)

amount_of_cpus_to_use = int(config['EXPERIMENTS']['amount_of_cpus'])
pool = mp.Pool(amount_of_cpus_to_use)

scenarios = config["EXPERIMENTS"]["scenarios"].split(",")
approach_names = config["EXPERIMENTS"]["approaches"].split(",")
amount_of_scenario_training_instances = int(
    config["EXPERIMENTS"]["amount_of_training_scenario_instances"])
tune_hyperparameters = bool(int(config["EXPERIMENTS"]["tune_hyperparameters"]))

for fold in range(1, 11):

    for scenario in scenarios:
        approaches = create_approach(approach_names)

        if len(approaches) < 1:
            logger.error("No approaches recognized!")
        for approach in approaches:
            metrics = list()
            metrics.append(Par10Metric())
            if approach.get_name() != 'oracle':
                metrics.append(NumberUnsolvedInstances(False))
                metrics.append(NumberUnsolvedInstances(True))
            logger.info("Submitted pool task for approach \"" +
                        str(approach.get_name()) + "\" on scenario: " + scenario)
            pool.apply_async(evaluate_scenario, args=(scenario, approach, metrics,
                                                      amount_of_scenario_training_instances, fold, config, tune_hyperparameters), callback=log_result)

            #evaluate_scenario(scenario, approach, metrics,
            #                  amount_of_scenario_training_instances, fold, config, tune_hyperparameters)

            print('Finished evaluation of fold')

pool.close()
pool.join()
logger.info("Finished all experiments.")
