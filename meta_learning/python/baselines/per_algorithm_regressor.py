import logging
import pandas as pd
import numpy as np
from matplotlib.pyplot import sci
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.base import clone
from .utils import impute_censored, distr_func
from aslib_scenario.aslib_scenario import ASlibScenario


class PerAlgorithmRegressor:

    def __init__(self, scikit_regressor=RandomForestRegressor(n_jobs=1, n_estimators=100), impute_censored=False):
        self.scikit_regressor = scikit_regressor
        self.logger = logging.getLogger("per_algorithm_regressor")
        self.logger.addHandler(logging.StreamHandler())
        self.trained_models = list()
        self.trained_imputers = list()
        self.trained_scalers = list()
        self.num_algorithms = 0
        self.algorithm_cutoff_time = -1
        self.impute_censored = impute_censored

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        print("Run fit on " + self.get_name() + " for fold " + str(fold))
        self.num_algorithms = len(scenario.algorithms)
        self.algorithm_cutoff_time = scenario.algorithm_cutoff_time

        for algorithm_id in range(self.num_algorithms):
            X_train, y_train = self.get_x_y(
                scenario, amount_of_training_instances, algorithm_id, fold)

            # impute missing values
            imputer = SimpleImputer()
            X_train = imputer.fit_transform(X_train)
            self.trained_imputers.append(imputer)

            # standardize feature values
            standard_scaler = StandardScaler()
            X_train = standard_scaler.fit_transform(X_train)
            self.trained_scalers.append(standard_scaler)

            model = clone(self.scikit_regressor)
            model.set_params(random_state=fold)

            if self.impute_censored:
                censored = y_train >= self.algorithm_cutoff_time
                model = impute_censored(
                    X_train, y_train, censored, model, distr_func, self.algorithm_cutoff_time)

            else:
                model.fit(X_train, y_train)

            self.trained_models.append(model)

        print("Finished training " + str(self.num_algorithms) +
              " models on " + str(amount_of_training_instances) + " instances.")

    def predict(self, features_of_test_instance, instance_id: int):
        predicted_risk_scores = list()

        for algorithm_id in range(self.num_algorithms):
            X_test = np.reshape(features_of_test_instance,
                                (1, len(features_of_test_instance)))

            imputer = self.trained_imputers[algorithm_id]
            X_test = imputer.transform(X_test)

            scaler = self.trained_scalers[algorithm_id]
            X_test = scaler.transform(X_test)

            model = self.trained_models[algorithm_id]

            prediction = model.predict(X_test)
            predicted_risk_scores.append(prediction)

        return np.asarray(predicted_risk_scores)

    def get_x_y(self, scenario: ASlibScenario, num_requested_instances: int, algorithm_id: int, fold: int):
        amount_of_training_instances = min(num_requested_instances,
                                           len(scenario.instances)) if num_requested_instances > 0 else len(
            scenario.instances)
        resampled_scenario_feature_data, resampled_scenario_performances = resample(scenario.feature_data,
                                                                                    scenario.performance_data,
                                                                                    n_samples=amount_of_training_instances,
                                                                                    random_state=fold)  # scenario.feature_data, scenario.performance_data #

        X_for_algorithm_id, y_for_algorithm_id = self.construct_dataset_for_algorithm_id(resampled_scenario_feature_data,
                                                                                         resampled_scenario_performances, algorithm_id,
                                                                                         scenario.algorithm_cutoff_time)

        return X_for_algorithm_id, y_for_algorithm_id

    def construct_dataset_for_algorithm_id(self, instance_features, performances, algorithm_id: int,
                                           algorithm_cutoff_time):
        performances_of_algorithm_with_id = performances.iloc[:, algorithm_id].to_numpy(
        ) if isinstance(performances, pd.DataFrame) else performances[:, algorithm_id]
        num_instances = len(performances_of_algorithm_with_id)

        if isinstance(instance_features, pd.DataFrame):
            instance_features = instance_features.to_numpy()

        # drop all instances for the respective algorithm that contain nan values
        nan_mask = np.isnan(performances_of_algorithm_with_id)
        instance_features = instance_features[~nan_mask]
        performances_of_algorithm_with_id = performances_of_algorithm_with_id[~nan_mask]

        return instance_features, performances_of_algorithm_with_id

    def get_name(self):
        name = ''
        if self.impute_censored:
            name += 'imputed_'

        name += 'per_algorithm_{}_regressor'.format(
            type(self.scikit_regressor).__name__)
        return name
