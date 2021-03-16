from aslib_scenario.aslib_scenario import ASlibScenario
import numpy as np
import logging


class SingleBestSolver:

    def __init__(self):
        self.logger = logging.getLogger("sbs")
        self.logger.addHandler(logging.StreamHandler())

        self.mean_train_runtimes = None

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        runtimes = scenario.performance_data.to_numpy()
        self.mean_train_runtimes = np.mean(runtimes, axis=0)

#        min_values = list()
#        for row in runtimes:
#            min_val = np.min(row)
#            if min_val < 50000:
#                min_values.append(min_val)
#        min_values = np.asarray(min_values)
#        print("AVG_RT: " + str(fold) + " . " + str(np.average(min_values)) + " . " + str(np.max(min_values)) + " . " + str(np.median(min_values)))

    def predict(self, features_of_test_instance, instance_id: int):
        return self.mean_train_runtimes

    def get_name(self):
        return "sbs"