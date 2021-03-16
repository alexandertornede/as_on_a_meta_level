import numpy as np


class NumberUnsolvedInstances:

    def __init__(self, with_feature_cost: bool):
        self.with_feature_cost = with_feature_cost


    def evaluate(self, gt_runtimes: np.ndarray, predicted_scores: np.ndarray, feature_cost: float, algorithm_cutoff_time: int):
        selected_algorithm_id  = np.argmin(predicted_scores)
        runtime_of_selected_algorithm = gt_runtimes[selected_algorithm_id]
        if self.with_feature_cost:
            runtime_of_selected_algorithm = runtime_of_selected_algorithm + feature_cost

        return 1 if runtime_of_selected_algorithm > algorithm_cutoff_time else 0

    def get_name(self):
        return "number_unsolved_instances_" + str(self.with_feature_cost)
