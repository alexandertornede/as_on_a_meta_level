import logging
import pickle
import sys
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression, mutual_info_regression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeRegressor
from sklearn.utils import resample
from sklearn.base import clone

from par_10_metric import Par10Metric
from .utils import impute_censored, distr_func
from aslib_scenario.aslib_scenario import ASlibScenario
from sklearn.pipeline import Pipeline


class PerAlgorithmRegressor:

    def __init__(self, scikit_regressor=RandomForestRegressor(n_jobs=1, n_estimators=100), impute_censored=False, feature_selection=None, feature_importances=False):
        self.scikit_regressor = scikit_regressor
        self.logger = logging.getLogger("per_algorithm_regressor")
        self.logger.addHandler(logging.StreamHandler())

        # attributes
        self.trained_models = list()
        self.trained_pipes = list()
        self.num_algorithms = 0
        self.algorithm_cutoff_time = -1
        self.metric = Par10Metric()
        self.feature_importances = feature_importances

        # setup
        self.impute_censored = impute_censored
        self.feature_selection = feature_selection

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        # set attributes
        self.num_algorithms = len(scenario.algorithms)
        self.algorithm_cutoff_time = scenario.algorithm_cutoff_time

        # train a regressor function per algorithm
        for algorithm_id in range(self.num_algorithms):
            X_train, y_train = self.get_x_y(
                scenario, amount_of_training_instances, algorithm_id, fold)

            # pipeline selection - with or without feature selection
            if self.feature_selection is None:
                pipe = Pipeline([('imputer', SimpleImputer()), ('standard_scaler', StandardScaler())])
            elif self.feature_selection == 'VarianceThreshold':
                pipe = Pipeline([('imputer', SimpleImputer()), ('standard_scaler', StandardScaler()), ('VarianceThreshold', VarianceThreshold(threshold=(.8 * (1 - .8))))])
            elif self.feature_selection == 'SelectKBest_f_regression':
                optimal_feature_number = self.calculate_optimal_feature_number('SelectKBest_f_regression', X_train, y_train, fold, scenario)
                pipe = Pipeline([('imputer', SimpleImputer()), ('standard_scaler', StandardScaler()), ('SelectKBest_f_regression', SelectKBest(f_regression, k=optimal_feature_number))])
            elif self.feature_selection == 'SelectKBest_mutual_info_regression':
                optimal_feature_number = self.calculate_optimal_feature_number('SelectKBest_mutual_info_regression', X_train, y_train, fold, scenario)
                pipe = Pipeline([('imputer', SimpleImputer()), ('standard_scaler', StandardScaler()), ('SelectKBest_mutual_info_regression', SelectKBest(mutual_info_regression, k=optimal_feature_number))])

            # apply the pipeline
            X_train = pipe.fit_transform(X_train, y_train)
            self.trained_pipes.append(pipe)

            # train the model (with/without stump/weight)
            model = clone(self.scikit_regressor)
            model.set_params(random_state=fold)

            if self.impute_censored:
                censored = y_train >= self.algorithm_cutoff_time
                model = impute_censored(
                    X_train, y_train, censored, model, distr_func, self.algorithm_cutoff_time)
            else:
                model.fit(X_train, y_train)
                if self.feature_importances:
                    self.save_feature_importance(model, scenario.scenario, len(X_train[0]))

            self.trained_models.append(model)

    def predict(self, features_of_test_instance, instance_id: int):
        predicted_risk_scores = list()

        for algorithm_id in range(self.num_algorithms):
            X_test = np.reshape(features_of_test_instance,
                                (1, len(features_of_test_instance)))

            # prepare features for prediction
            X_test = self.trained_pipes[algorithm_id].transform(X_test)

            model = self.trained_models[algorithm_id]

            prediction = model.predict(X_test)
            predicted_risk_scores.append(prediction)

        return np.asarray(predicted_risk_scores)

    def calculate_optimal_feature_number(self, feature_selector, X_data, y_data, fold: int, scenario: ASlibScenario):
        # create a 9/10 train - 1/10 test split
        X_train, y_train, X_test, y_test = self.create_fold(1, X_data, y_data)
        best_score = 1000000
        feature_number = len(X_train[0])
        optimal_number = 1
        feature_cost_data = scenario.feature_cost_data.to_numpy() if scenario.feature_cost_data is not None else None

        # add 10 or 11 steps for the optimal feature number
        if feature_number > 9:
            steps = np.arange(start=1, stop=feature_number, step=int(feature_number / 10))
        else:
            steps = np.arange(start=1, stop=feature_number, step=1)

        # train and test the feature numbers with the split
        for number_of_features in steps:
            if feature_selector == 'SelectKBest_f_regression':
                pipe = Pipeline([('imputer', SimpleImputer()), ('standard_scaler', StandardScaler()), ('SelectKBest_f_regression', SelectKBest(f_regression, k=number_of_features))])
            else:
                pipe = Pipeline([('imputer', SimpleImputer()), ('standard_scaler', StandardScaler()), ('SelectKBest_mutual_info_regression', SelectKBest(mutual_info_regression, k=number_of_features))])

            X_train_validation = pipe.fit_transform(X_train, y_train)
            X_test_validation = pipe.transform(X_test)

            model = clone(self.scikit_regressor)
            model.set_params(random_state=fold)
            model.fit(X_train_validation, y_train)

            performance_measure = 0
            for instance_id, x_test in enumerate(X_test_validation):
                # TODO: Only works for fold 1!
                accumulated_feature_time = 0
                if scenario.feature_cost_data is not None:
                    feature_time = feature_cost_data[instance_id]
                    accumulated_feature_time = np.sum(feature_time)

                predicted_scores = model.predict(x_test.reshape(1, -1))
                performance_measure = performance_measure + self.metric.evaluate(y_test, predicted_scores,
                                                                                 accumulated_feature_time,
                                                                                 scenario.algorithm_cutoff_time)
            if best_score > performance_measure:
                best_score = performance_measure
                optimal_number = number_of_features
            ("Score", performance_measure, "with", number_of_features, "features.")

        return optimal_number

    def create_fold(self, fold, X_data, y_data):
        fold_lenght = int(len(X_data) / 10)
        if fold == 1:
            X_test = X_data[:fold * fold_lenght]
            y_test = y_data[:fold * fold_lenght]

            X_train = X_data[fold * fold_lenght:]
            y_train = y_data[fold * fold_lenght:]
        elif fold == 10:
            X_test = X_data[(fold - 1) * fold_lenght:]
            y_test = y_data[(fold - 1) * fold_lenght:]

            X_train = X_data[:(fold - 1) * fold_lenght]
            y_train = y_data[:(fold - 1) * fold_lenght]
        else:
            X_test = X_data[(fold - 1) * fold_lenght:fold * fold_lenght]
            y_test = y_data[(fold - 1) * fold_lenght:fold * fold_lenght]

            X_train = X_data[:(fold - 1) * fold_lenght] + X_data[fold * fold_lenght:]
            y_train = y_data[:(fold - 1) * fold_lenght] + y_data[fold * fold_lenght:]
        return X_train, y_train, X_test, y_test


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

    # save base learner for later use
    def save_feature_importance(self, base_learner, scenario_name, num_features):
        importances = base_learner.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Print the feature ranking
        print("Feature ranking:")

        file_name = 'feature_importance/' + scenario_name
        with open(file_name, 'ab') as f:
            pickle.dump((-1, num_features), f)
            for i in indices:
                data = (i, importances[i])
                print(data)
                pickle.dump(data, f)

    def get_name(self):
        name = ''
        if self.impute_censored:
            name += 'imputed_'
        name += 'per_algorithm_{}_regressor'.format(
            type(self.scikit_regressor).__name__)
        if self.feature_selection is not None:
            name += self.feature_selection
        return name
