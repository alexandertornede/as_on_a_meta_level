from aslib_scenario.aslib_scenario import ASlibScenario
from ax import *
import copy
import numpy as np
from par_10_metric import Par10Metric
from evaluation_of_train_test_split import evaluate_train_test_split
import logging

class HyperParameterOptimizer:

    def __init__(self):
        self.logger = logging.getLogger("hyper_parameter_optimizer")
        self.logger.addHandler(logging.StreamHandler())
        self.scenario = None
        self.approach = None

    def evaluate_parameters(self, parameterization, weight=None):
        self.logger.debug("Evaluating parametrization " + str(parameterization))
        num_splits = 2
        scenario_copy = copy.copy(self.scenario)
        scenario_copy.create_cv_splits(num_splits)
        evaluation_results = list()
        for split in range(1, num_splits + 1):
            approach_for_split = copy.deepcopy(self.approach)
            approach_for_split.set_parameters(parameterization)
            metrics = [Par10Metric()]
            evaluation_result = evaluate_train_test_split(scenario_copy, approach_for_split, metrics, split, len(scenario_copy.instances))
            evaluation_results.append(evaluation_result[0])
            self.logger.debug("Peformance of parametrization " + str(parameterization) + ":" + str(evaluation_result[0]))

        evaluation_results = np.asarray(evaluation_results)
        self.logger.debug("Final peformance of parametrization " + str(parameterization) + ":" + str(np.average(evaluation_results)))

        return {"par10": (np.average(evaluation_results), np.std(evaluation_results))}

    def optimize(self, scenario: ASlibScenario, approach):
        self.scenario = scenario
        self.approach = approach

        self.logger.debug("Optimizing parameters of approach " + str(approach.get_name()) + " on scenario " + str(scenario.scenario))
        search_space = SearchSpace(
            parameters=[
                RangeParameter(
                    name="n_estimators", parameter_type=ParameterType.INT, lower=10, upper=1000
                ),
                RangeParameter(
                    name="min_samples_split", parameter_type=ParameterType.INT, lower=1, upper=100
                ),
                RangeParameter(
                    name="min_samples_leaf", parameter_type=ParameterType.INT, lower=1, upper=100
                ),
                RangeParameter(
                    name="min_weight_fraction_leaf", parameter_type=ParameterType.FLOAT, lower=0.0, upper=0.3
                ),
                ChoiceParameter(
                    name="max_features", parameter_type=ParameterType.STRING, values=["auto", "sqrt", "log2"]
                ),
                FixedParameter(
                    name="bootstrap", parameter_type=ParameterType.BOOL, value=True
                ),
                ChoiceParameter(
                    name="oob_score", parameter_type=ParameterType.BOOL, values=[True, False]
                )
            ]
        )

        exp = SimpleExperiment(
            name="find_survival_forest_hyper_params",
            search_space=search_space,
            evaluation_function=self.evaluate_parameters,
            objective_name="par10",
            minimize=True
        )

        sobol = Models.SOBOL(exp.search_space)
        for i in range(5):
            exp.new_trial(generator_run=sobol.gen(1))

        best_arm = None
        for i in range(15):
            gpei = Models.GPEI(experiment=exp, data=exp.eval())
            generator_run = gpei.gen(1)
            best_arm, _ = generator_run.best_arm_predictions
            exp.new_trial(generator_run=generator_run)

        self.logger.debug("Finished optimization of parameters of approach " + str(approach) + " on scenario " + str(scenario))

        return best_arm.parameters
