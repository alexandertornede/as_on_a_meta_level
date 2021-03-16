import logging
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

from survival_approaches.survival_forests.surrogate import SurrogateSurvivalForest
from baselines.isac import ISAC
from baselines.multiclass_algorithm_selector import MultiClassAlgorithmSelector
from baselines.per_algorithm_regressor import PerAlgorithmRegressor
from aslib_scenario.aslib_scenario import ASlibScenario
from baselines.satzilla11 import SATzilla11
from baselines.sunny import SUNNY
import copy
import pandas as pd
import sys

from ensembles.validation import split_scenario
from pre_compute.pickle_loader import load_pickle


class StackingH2O:

    def __init__(self, base_learner=None, meta_learner_type='per_algorithm_regressor', pre_computed=False):
        self.logger = logging.getLogger("stacking")
        self.logger.addHandler(logging.StreamHandler())

        # parameters
        self.meta_learner_type = meta_learner_type
        self.pre_computed = pre_computed
        self.base_learner_type = base_learner


        # attributes
        self.meta_learner = None
        self.base_learners = list()
        self.num_algorithms = 0
        self.scenario_name = ''
        self.fold = 0
        self.predictions = list()
        self.algorithm_selection_algorithm = False
        self.pipe = None

    def create_base_learner(self):
        self.base_learners = list()
        if 1 in self.base_learner_type:
            self.base_learners.append(PerAlgorithmRegressor())
        if 2 in self.base_learner_type:
            self.base_learners.append(SUNNY())
        if 3 in self.base_learner_type:
            self.base_learners.append(ISAC())
        if 4 in self.base_learner_type:
            self.base_learners.append(SATzilla11())
        if 5 in self.base_learner_type:
            self.base_learners.append(SurrogateSurvivalForest(criterion='Expectation'))
        if 6 in self.base_learner_type:
            self.base_learners.append(SurrogateSurvivalForest(criterion='PAR10'))
        if 7 in self.base_learner_type:
            self.base_learners.append(MultiClassAlgorithmSelector())

    def fit(self, scenario: ASlibScenario, fold: int, amount_of_training_instances: int):
        self.create_base_learner()
        self.scenario_name = scenario.scenario
        self.fold = fold
        self.num_algorithms = len(scenario.algorithms)
        num_instances = len(scenario.instances)
        feature_data = scenario.feature_data.to_numpy()
        performance_data = scenario.performance_data.to_numpy()
        new_feature_data = np.zeros((num_instances, self.num_algorithms * len(self.base_learners)))

        for learner_index, base_learner in enumerate(self.base_learners):

            instance_counter = 0

            predictions = np.zeros((num_instances, self.num_algorithms))

            if self.pre_computed:
                predictions = load_pickle(filename='predictions/cross_validation_' + base_learner.get_name() + '_' + scenario.scenario + '_' + str(fold))
            else:
                for sub_fold in range(1, 11):
                    test_scenario, training_scenario = split_scenario(scenario, sub_fold, num_instances)

                    # train base learner
                    base_learner.fit(training_scenario, fold, amount_of_training_instances)

                    # create new feature data
                    for instance_number in range(instance_counter, instance_counter + len(test_scenario.instances)):
                        prediction = base_learner.predict(feature_data[instance_number], instance_number)
                        predictions[instance_number] = prediction.flatten()

                    instance_counter = instance_counter + len(test_scenario.instances)

            for i in range(num_instances):
                for alo_num in range(self.num_algorithms):
                    new_feature_data[i][alo_num + self.num_algorithms * learner_index] = predictions[i][alo_num]

        if self.pre_computed:
            for base_learner in self.base_learners:
                self.predictions.append(load_pickle(filename='predictions/' + base_learner.get_name() + '_' + scenario.scenario + '_' + str(fold)))
        else:
            self.create_base_learner()
            for base_learner in self.base_learners:
                base_learner.fit(scenario, fold, amount_of_training_instances)

        # add predictions to the features of the instances
        new_feature_data = pd.DataFrame(new_feature_data, index=scenario.feature_data.index, columns=np.arange(self.num_algorithms * len(self.base_learners)))
        new_feature_data = pd.concat([scenario.feature_data, new_feature_data], axis=1, sort=False)
        scenario.feature_data = new_feature_data

        # meta learner training with or without feature selection
        if self.meta_learner_type == 'per_algorithm_regressor':
            self.meta_learner = PerAlgorithmRegressor()
            self.algorithm_selection_algorithm = True
        elif self.meta_learner_type == 'SUNNY':
            self.meta_learner = SUNNY()
            self.algorithm_selection_algorithm = True
        elif self.meta_learner_type == 'ISAC':
            self.meta_learner = ISAC()
            self.algorithm_selection_algorithm = True
        elif self.meta_learner_type == 'SATzilla-11':
            self.meta_learner = SATzilla11()
            self.algorithm_selection_algorithm = True
        elif self.meta_learner_type == 'multiclass':
            self.meta_learner = MultiClassAlgorithmSelector()
            self.algorithm_selection_algorithm = True
        elif self.meta_learner_type == 'Expectation':
            self.meta_learner = SurrogateSurvivalForest(criterion='Expectation')
            self.algorithm_selection_algorithm = True
        elif self.meta_learner_type == 'RandomForest':
            self.meta_learner = DecisionTreeClassifier()
        elif self.meta_learner_type == 'RandomForest':
            self.meta_learner = RandomForestClassifier()

        if self.algorithm_selection_algorithm:
            self.meta_learner.fit(scenario, fold, amount_of_training_instances)
        else:
            label_performance_data = [np.argmin(x) for x in performance_data]

            self.pipe = Pipeline([('imputer', SimpleImputer()), ('standard_scaler', StandardScaler())])
            X_train = self.pipe.fit_transform(scenario.feature_data.to_numpy(), label_performance_data)

            self.meta_learner.fit(X_train, label_performance_data)

    def predict(self, features_of_test_instance, instance_id: int):
        # get all predictions from the base learners
        new_feature_data = np.zeros(self.num_algorithms * len(self.base_learners))

        for learner_index, base_learner in enumerate(self.base_learners):
            # create new feature data
            if self.pre_computed:
                prediction = self.predictions[learner_index][str(features_of_test_instance)]
            else:
                prediction = base_learner.predict(features_of_test_instance, instance_id).flatten()

            for alo_num in range(self.num_algorithms):
                new_feature_data[alo_num + self.num_algorithms * learner_index] = prediction[alo_num]

        features_of_test_instance = np.concatenate((features_of_test_instance, new_feature_data), axis=0)

        # final prediction
        if self.algorithm_selection_algorithm:
            return self.meta_learner.predict(features_of_test_instance, instance_id)
        else:
            X_test = self.pipe.transform(features_of_test_instance.reshape(1, -1))
            final_prediction = np.ones(self.num_algorithms)
            final_prediction[self.meta_learner.predict(X_test)] = 0
            return final_prediction

    def get_name(self):
        name = "stackingh2o_" + "_" + str(self.base_learner_type).replace('[', '').replace(']', '').replace(', ', '_') + self.meta_learner_type

        return name
