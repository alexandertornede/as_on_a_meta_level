import numpy as np


class PerformanceMetric:
    def evaluate(self, gt_runtimes: np.ndarray, predicted_scores: np.ndarray, feature_cost: float,
                 algorithm_cutoff_time: int):
        selected_algorithm_id = np.argmin(predicted_scores)
        negative_performance_of_selected_algorithm = gt_runtimes[selected_algorithm_id]
        if feature_cost > algorithm_cutoff_time:
            negative_performance_of_selected_algorithm = 1
        # make performance positive again such that the result is interpretable    
        return (-1) * negative_performance_of_selected_algorithm

    def get_name(self):
        return "performance"
