import numpy as np


class Par10Metric:

    def evaluate(self, gt_runtimes: np.ndarray, predicted_scores: np.ndarray, feature_cost: float, algorithm_cutoff_time: int):
        #print("gt: " + str(gt_runtimes))
        #print("pred: " + str(predicted_scores))
        selected_algorithm_id  = np.argmin(predicted_scores)
        runtime_of_selected_algorithm = gt_runtimes[selected_algorithm_id]
        runtime_including_feature_costs = runtime_of_selected_algorithm + feature_cost

        if runtime_including_feature_costs > algorithm_cutoff_time:
            runtime_including_feature_costs = (algorithm_cutoff_time * 10)

        return runtime_including_feature_costs

    def get_name(self):
        return "par10"
