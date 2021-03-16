from aslib_scenario.aslib_scenario import ASlibScenario
import pandas as pd
import numpy as np
from matplotlib.pyplot import sci
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import logging
from sklearn.base import clone
#import matplotlib.pyplot as plt
#import seaborn as sns


class MultiClassAlgorithmSelector:

    def __init__(self, scikit_classifier=RandomForestClassifier(n_jobs=1, n_estimators=100)):
        self.scikit_classifier= scikit_classifier
        self.logger = logging.getLogger("multiclass_algorithm_selector")
        self.logger.addHandler(logging.StreamHandler())
        self.trained_model = None
        self.imputer = None
        self.standard_scaler = None
        self.num_algorithms = 0
        self.algorithm_cutoff_time = -1;

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        print("Run fit on " + self.get_name() + " for fold " + str(fold))
        self.num_algorithms = len(scenario.algorithms)
        self.algorithm_cutoff_time = scenario.algorithm_cutoff_time

        X_train, y_train = self.get_x_y(scenario, amount_of_training_instances, fold)

        # impute missing values
        self.imputer = SimpleImputer()
        X_train = self.imputer.fit_transform(X_train)

        # standardize feature values
        self.scaler = StandardScaler()
        X_train = self.scaler.fit_transform(X_train)

        self.trained_model = clone(self.scikit_classifier)
        self.trained_model.set_params(random_state=fold)
        self.trained_model.fit(X_train, y_train)

        print("Finished training " + str(self.num_algorithms) + " models on " + str(amount_of_training_instances) + " instances.")

    def predict(self, features_of_test_instance, instance_id: int):
        X_test = np.reshape(features_of_test_instance, (1, len(features_of_test_instance)))
        X_test = self.imputer.transform(X_test)
        X_test = self.scaler.transform(X_test)

        prediction = self.trained_model.predict(X_test)[0]

        predicted_performances = np.empty(self.num_algorithms)
        predicted_performances.fill(1)
        #make sure the predicted algorithm has the lowest, i.e. best score
        predicted_performances[prediction] = 0

        return predicted_performances

    def get_x_y(self, scenario: ASlibScenario, num_requested_instances: int, fold: int):
        amount_of_training_instances = min(num_requested_instances,
                                           len(scenario.instances)) if num_requested_instances > 0 else len(
            scenario.instances)
        resampled_scenario_feature_data, resampled_scenario_performances = resample(scenario.feature_data,
                                                                                    scenario.performance_data,
                                                                                    n_samples=amount_of_training_instances,
                                                                                    random_state=fold)  # scenario.feature_data, scenario.performance_data #

        X, y = self.construct_dataset(resampled_scenario_feature_data, resampled_scenario_performances)

        return X, y


    def construct_dataset(self, instance_features, performances):
        performances = performances.iloc[:, :].to_numpy() if isinstance(performances, pd.DataFrame) else performances[:, :]

        # ignore all unsolvable training instances 
        nan_mask = np.all(np.isnan(performances), axis=1)
        instance_features = instance_features[~nan_mask]
        performances = performances[~nan_mask]       

        num_instances = len(performances)
        best_algorithm_ids = list()
        for i in range(0, num_instances):
            min_runtime = np.nanmin(performances[i])
            best_algorithm_id = np.nonzero(performances[i] == min_runtime)[0][0]

            best_algorithm_id = np.argmin(performances[i])
            best_algorithm_ids.append(best_algorithm_id)

        if isinstance(instance_features, pd.DataFrame):
            instance_features = instance_features.to_numpy()

        return instance_features, np.asarray(best_algorithm_ids)

    def get_name(self):
        return "multiclass_algorithm_selector"