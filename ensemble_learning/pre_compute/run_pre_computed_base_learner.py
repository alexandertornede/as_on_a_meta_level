from aslib_scenario.aslib_scenario import ASlibScenario

from pre_compute.pickle_loader import load_pickle


class RunPreComputedBaseLearner:

    def __init__(self, algorithm):
        self.algorithm = algorithm
        self.num_algorithms = 0
        self.base_learner = None
        self.scenario_name = ''
        self.fold = 0
        self.predictions = None

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        self.num_algorithms = len(scenario.algorithms)
        self.scenario_name = scenario.scenario
        self.fold = fold

        self.predictions = load_pickle('predictions/' + self.algorithm + '_' + self.scenario_name + '_' + str(self.fold))



    def predict(self, features_of_test_instance, instance_id: int):
        return self.predictions[str(features_of_test_instance)]

    def get_name(self):
        name = self.algorithm
        return name
