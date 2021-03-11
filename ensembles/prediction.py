import numpy as np
from scipy.stats import rankdata


def predict_with_ranking(features_of_test_instance, instance_id: int, num_algorithms: int, trained_models, weights=None, performance_rank=False, rank_method='average', pre_computed_predictions=None):
    # use all predictions (ranking) of the base learner
    predictions = np.zeros((num_algorithms, 1))
    for i, model in enumerate(trained_models):
        if pre_computed_predictions is not None:
            prediction = pre_computed_predictions[i][str(features_of_test_instance)]
        else:
            prediction = model.predict(features_of_test_instance, instance_id)

        # rank output from the base learner (best algorithm gets rank 1 - worst algorithm gets rank len(algorithms))
        if weights:
            if performance_rank:
                predictions = predictions.reshape(num_algorithms) + weights[i] * (prediction / len(trained_models)).reshape(num_algorithms)
            else:
                predictions = predictions.reshape(num_algorithms) + weights[i] * rankdata(prediction, method=rank_method).reshape(num_algorithms)
        else:
            if performance_rank:
                predictions = predictions.reshape(num_algorithms) + (prediction / len(trained_models)).reshape(num_algorithms)
            else:
                predictions = predictions.reshape(num_algorithms) + rankdata(prediction, method=rank_method)

    return predictions / sum(predictions)