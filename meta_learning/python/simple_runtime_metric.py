import numpy as np


class RuntimeMetric:

    def evaluate(self, gt_runtimes: np.ndarray, predicted_scores: np.ndarray, feature_cost: float, algorithm_cutoff_time: int):
        selected_algorithm_id  = np.argmin(predicted_scores)
        runtime_of_selected_algorithm = gt_runtimes[selected_algorithm_id]
        runtime_without_feature_costs = runtime_of_selected_algorithm

        if runtime_without_feature_costs > algorithm_cutoff_time:
            runtime_without_feature_costs = algorithm_cutoff_time

        return runtime_without_feature_costs

    def get_name(self):
        return "runtime"
