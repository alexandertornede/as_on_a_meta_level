from aslib_scenario.aslib_scenario import ASlibScenario
import numpy as np
import logging


class SingleBestSolverWithFeatureCosts:

    def __init__(self):
        self.logger = logging.getLogger("sbs_with_feature_costs")
        self.logger.addHandler(logging.StreamHandler())

        self.mean_train_runtimes = None

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        runtimes = scenario.performance_data.to_numpy()
        self.mean_train_runtimes = np.mean(runtimes, axis=0)


    def predict(self, features_of_test_instance, instance_id: int):
        return self.mean_train_runtimes

    def get_name(self):
        return "sbs_with_feature_costs"