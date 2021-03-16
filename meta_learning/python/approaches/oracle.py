from aslib_scenario.aslib_scenario import ASlibScenario
import logging


class Oracle:

    def __init__(self):
        self.logger = logging.getLogger("oracle")
        self.logger.addHandler(logging.StreamHandler())

        self.performances = None
        self.feature_data = None

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        self.performances = scenario.performance_data.to_numpy()
        self.feature_data = scenario.feature_data.to_numpy()

    def predict(self, features_of_test_instance, instance_id: int):
        return self.performances[instance_id]

    def get_name(self):
        return "oracle"