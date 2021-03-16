import numpy as np


class Par10SchedulingMetric:

    def __init__(self, amount_of_algrithms_in_schedule: int):
        self.amount_of_algorithms_in_schedule = amount_of_algrithms_in_schedule

    def evaluate(self, gt_runtimes: np.ndarray, predicted_scores: np.ndarray, feature_cost: float, algorithm_cutoff_time: int):
        countable_amount_of_algorithms = min(self.amount_of_algorithms_in_schedule, len(gt_runtimes)) if self.amount_of_algorithms_in_schedule > 0 else len(gt_runtimes)

        per_algorithm_runtime_threshold = (algorithm_cutoff_time / float(countable_amount_of_algorithms))
        sorted_indices_according_to_predicted_scores = np.argsort(predicted_scores)
        accumulated_runtime = feature_cost
        solved = False

        while not solved:
            for i in range(0, countable_amount_of_algorithms):
                algorithm_id_of_ith_choice = sorted_indices_according_to_predicted_scores[i]
                runtime_of_ith_choice = gt_runtimes[algorithm_id_of_ith_choice]

                if runtime_of_ith_choice <= per_algorithm_runtime_threshold:
                    accumulated_runtime += runtime_of_ith_choice
                    solved = True
                    break
                else:
                    accumulated_runtime += per_algorithm_runtime_threshold

            if solved or accumulated_runtime > algorithm_cutoff_time:
                break

            per_algorithm_runtime_threshold = per_algorithm_runtime_threshold * 2

        if accumulated_runtime > algorithm_cutoff_time or not solved:
            accumulated_runtime = (algorithm_cutoff_time * 10)

        return accumulated_runtime

    def get_name(self):
        return "par10_scheduling_" + str(self.amount_of_algorithms_in_schedule)
