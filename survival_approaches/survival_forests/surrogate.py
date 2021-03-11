import logging
import warnings
import pandas as pd
import numpy as np
from aslib_scenario.aslib_scenario import ASlibScenario
from sksurv.ensemble import RandomSurvivalForest
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from ax import optimize
from operator import itemgetter


class SurrogateSurvivalForest:

    def __init__(self, criterion='Expectation'):
        self.logger = logging.getLogger("" + criterion + "SurvivalForest")
        self.logger.addHandler(logging.StreamHandler())
        self.num_algorithms = 0
        self.algorithm_cutoff_time = -1;

        self.n_estimators = 100 #1000
        self.min_samples_split = 10
        self.min_samples_leaf = 15
        self.min_weight_fraction_leaf = 0.0
        self.max_features = "sqrt"
        self.bootstrap = True
        self.oob_score = False
        self.criterion = criterion
        
        
    def fit(self, scenario: ASlibScenario, fold: int, num_instances: int):
        self.num_algorithms = len(scenario.algorithms)
        self.algorithm_cutoff_time = scenario.algorithm_cutoff_time
        np.random.seed(fold)
        
        num_instances = min(num_instances, len(scenario.instances)) if num_instances > 0 else len(scenario.instances)
        features, performances = resample(scenario.feature_data, scenario.performance_data, n_samples=num_instances, random_state=fold)
        features = features.to_numpy()
        performances = performances.to_numpy()
        
        if self.criterion == 'Expectation':
            self.risk_func = lambda x: x
            self.imputer, self.scaler, self.models = self.fit_regressors(features, performances, random_state=fold)

        elif self.criterion == 'Polynomial':
            self.risk_func = lambda x: x**2
            self.imputer, self.scaler, self.models = self.fit_regressors(features, performances, random_state=fold)
        
        elif self.criterion == 'Exponential':
            warnings.filterwarnings('ignore')
            alpha = 1.0
            self.risk_func = lambda x: np.minimum((-1) * alpha * np.log(1.0 - x), 3.0)
            self.imputer, self.scaler, self.models = self.fit_regressors(features, performances, random_state=fold)

        elif self.criterion == 'PAR10':
            # PAR10 surrogate for normalized values between [0,1]
            self.risk_func = np.vectorize(lambda x: x if x < 1.0 else 10.0)
            self.imputer, self.scaler, self.models = self.fit_regressors(features, performances, random_state=fold)

        elif self.criterion == 'GridSearch':
            warnings.filterwarnings('ignore')
            
            # train model to later optimize surrogate hyperparameters against
            abs_split = np.size(features, axis=0)
            instance_idx = np.random.choice(abs_split, size=int(0.7 * abs_split), replace=False)
            X_train = features[instance_idx]
            X_val = np.delete(features, instance_idx, axis=0)
            Y_train = performances[instance_idx]
            Y_val = np.delete(performances, instance_idx, axis=0)
            val_imputer, val_scaler, val_models = self.fit_regressors(X_train, Y_train, random_state=fold)
            
            # obtain predictions from the validation model
            event_times, survival_functions = self.predict_survival_functions(X_val, val_imputer, val_scaler, val_models)
            
            # normalize event times to [0,1] interval (prevents numerical issues for exponential)
            for alg_id in range(self.num_algorithms):
                event_times[alg_id] /= self.algorithm_cutoff_time
            
            # search for best parametrization of risk function
            alpha_domain = np.linspace(1.0, 5.0, 21)
            results = [self.evaluate_surrogate(lambda x: x**alpha, event_times, survival_functions, Y_val) for alpha in alpha_domain]
            poly_idx, poly_val = min(enumerate(results), key=itemgetter(1))
            poly_alpha = alpha_domain[poly_idx]
            
            alpha_domain = np.linspace(1.0, 0.0, 20, endpoint=False)
            threshold = 5.0
            results = [self.evaluate_surrogate(lambda x: np.minimum((-1) * alpha * np.log(1.0 - x), threshold), event_times, survival_functions, Y_val) for alpha in alpha_domain]
            exp_idx, exp_val = min(enumerate(results), key=itemgetter(1))
            exp_alpha = alpha_domain[exp_idx]
            
            if poly_val <= exp_val:
                self.risk_func = lambda x: x**poly_alpha
            
            else:
                self.risk_func = lambda x: np.minimum((-1) * exp_alpha * np.log(1.0 - x), threshold)

            self.imputer, self.scaler, self.models = self.fit_regressors(features, performances, random_state=fold)

        else:
            raise ValueError('Unknown decision criterion')
        
    def evaluate_surrogate(self, risk_func, event_times, survival_functions, performances):
        num_inst = np.size(survival_functions[0], axis=0)
        risk = np.zeros(shape=(num_inst, self.num_algorithms))
        
        for alg_id in range(self.num_algorithms):
            # scale event times to [0,1] interval and compute expected risk
            event_times[alg_id] /= self.algorithm_cutoff_time
            risk[:, alg_id] = np.sum(survival_functions[alg_id] * np.diff(risk_func(event_times[alg_id])), axis=1)
        
        # compute respective algorithm selections for each instance and their consequential performance
        risk = np.argmin(risk, axis=1)
        risk = performances[np.arange(risk.size), risk]

        return np.mean(risk), np.std(risk)
        
        
    def predict(self, features, instance_id: int):
        assert(features.ndim == 1), 'Must be 1-dimensional'
        
        event_times = []
        survival_functions = []
        
        for alg_id in range(self.num_algorithms):
            X_test = np.reshape(features, (1, -1))
            X_test = self.imputer[alg_id].transform(X_test)
            X_test = self.scaler[alg_id].transform(X_test)
            event_times.append(self.models[alg_id].event_times_)
            survival_functions.append(self.models[alg_id].predict_survival_function(X_test)[0])
            
        for alg_id in range(self.num_algorithms):
            event_times[alg_id] = np.append(0.0, event_times[alg_id])
            event_times[alg_id] = np.append(event_times[alg_id], self.algorithm_cutoff_time)
            survival_functions[alg_id] = np.append(1.0, survival_functions[alg_id])
            
        expected_risk = np.zeros(self.num_algorithms)

        for alg_id in range(self.num_algorithms):
            event_times[alg_id] = event_times[alg_id] / self.algorithm_cutoff_time
            expected_risk[alg_id] = np.sum(survival_functions[alg_id] * np.diff(self.risk_func(event_times[alg_id])))

        return expected_risk
            
            
    def predict_survival_functions(self, features, imputer, scaler, models):
        
        assert(features.ndim == 2), 'Must be 2-dimensional'
        
        num_inst = np.size(features, axis=0)
        event_times = [model.event_times_ for model in models]
        survival_functions = [np.empty(shape=(num_inst, len(event_times[alg_id]) + 1)) for alg_id in range(self.num_algorithms)]
            
        for alg_id in range(self.num_algorithms):
                X_test = imputer[alg_id].transform(features)
                X_test = scaler[alg_id].transform(X_test)
                predictions = models[alg_id].predict_survival_function(X_test)
                tmp = np.full(fill_value=1.0, shape=(np.size(predictions, axis=0), 1))
                survival_functions[alg_id] = np.hstack([tmp, predictions])
                event_times[alg_id] = np.append(0.0, event_times[alg_id])
                event_times[alg_id] = np.append(event_times[alg_id], self.algorithm_cutoff_time)

        return event_times, survival_functions


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
        # get runtimes of algorithm 
        performances_of_algorithm_with_id = performances.iloc[:, algorithm_id].to_numpy() if isinstance(performances, pd.DataFrame) else performances[:, algorithm_id]
        num_instances = len(performances_of_algorithm_with_id)
        
        # for each instance determine whether it was finished before cutoff; also set PAR10 values
        finished_before_timeout = np.empty(num_instances, dtype=bool)
        for i in range(0, len(performances_of_algorithm_with_id)):
            finished_before_timeout[i] = True if (
                    performances_of_algorithm_with_id[i] < algorithm_cutoff_time) else False
     
        # for each instance build target, consisting of (censored, runtime)
        status_and_performance_of_algorithm_with_id = np.empty(dtype=[('cens', np.bool), ('time', np.float)],
                                                               shape=instance_features.shape[0])
        status_and_performance_of_algorithm_with_id['cens'] = finished_before_timeout
        status_and_performance_of_algorithm_with_id['time'] = performances_of_algorithm_with_id

        if isinstance(instance_features, pd.DataFrame):
            instance_features = instance_features.to_numpy()

        return instance_features, status_and_performance_of_algorithm_with_id.T

    def get_name(self):
        return "" + self.criterion + "_algorithm_survival_forest"
    
    def fit_thresh_regressor(self, X_train, y_train, random_state):
        thresh_imputer = SimpleImputer()
        X_train = thresh_imputer.fit_transform(X_train)
        thresh_scaler = StandardScaler()
        X_train = thresh_scaler.fit_transform(X_train)
            
        thresh_regressor = RandomForestRegressor(n_estimators=self.n_estimators,
                                 min_samples_split=self.min_samples_split,
                                 min_samples_leaf=self.min_samples_leaf,
                                 min_weight_fraction_leaf = self.min_weight_fraction_leaf,
                                 max_features=self.max_features,
                                 bootstrap = self.bootstrap,
                                 oob_score= self.oob_score,
                                 n_jobs=1,
                                 random_state=random_state)

        thresh_regressor.fit(X_train, y_train)
        return thresh_imputer, thresh_scaler, thresh_regressor

    def fit_regressors(self, features, performances, random_state):
        imputer = [SimpleImputer() for _ in range(self.num_algorithms)]
        scaler = [StandardScaler() for _ in range(self.num_algorithms)]
        models = [RandomSurvivalForest(n_estimators=self.n_estimators,
                                 min_samples_split=self.min_samples_split,
                                 min_samples_leaf=self.min_samples_leaf,
                                 min_weight_fraction_leaf = self.min_weight_fraction_leaf,
                                 max_features=self.max_features,
                                 bootstrap = self.bootstrap,
                                 oob_score= self.oob_score,
                                 n_jobs=1,
                                 random_state=random_state) for _ in range(self.num_algorithms)]

        for alg_id in range(self.num_algorithms):
            # prepare survival forest dataset and split the data accordingly
            X_train, Y_train = self.construct_dataset_for_algorithm_id(features, performances, alg_id, self.algorithm_cutoff_time)            
            X_train = imputer[alg_id].fit_transform(features)
            X_train = scaler[alg_id].fit_transform(X_train)
            models[alg_id].fit(X_train, Y_train)
        
        return imputer, scaler, models
    
    def event_time_expectation(self, overall_events, imputer, scaler, models, features, performances):
        expected_values = np.zeros(shape=(self.num_algorithms, np.size(features, axis=0), len(overall_events)))

        for alg_id in range(self.num_algorithms):
            X_train = imputer[alg_id].transform(features)
            X_train = scaler[alg_id].transform(X_train)

            # get model event times and predicted survival functions for all training instances
            model = models[alg_id]
            event_times = model.event_times_
            survival_functions = model.predict_survival_function(X_train)

            # preprocess event_times and survival_functions
            event_times = np.append(0.0, event_times)
            event_times = np.append(event_times, self.algorithm_cutoff_time)             
            survival_functions = np.hstack([np.broadcast_to(1.0, shape=(np.size(survival_functions, axis=0), 1)), survival_functions])   
            
            tmp_event = 0.0
            for event_idx, event in enumerate(overall_events):
                idx = np.argmax(event_times[1:] >= event)
                expected_values[alg_id, :, event_idx] = survival_functions[:, idx] * (event - tmp_event)
                tmp_event = event

        expected_values = np.flip(np.cumsum(np.flip(expected_values, axis=2), axis=2), axis=2)
        expected_values = np.argmin(expected_values, axis=0)
        return np.take_along_axis(performances, expected_values, axis=1)
    
        
    def scaled_threshold_expectation(self, alpha_domain, imputer, scaler, models, features, performances, lower=False):
        t_perf = np.zeros(shape=(self.num_algorithms, np.size(features, axis=0), np.size(alpha_domain)))

        for alg_id in range(self.num_algorithms):
            X_train = imputer[alg_id].transform(features)
            X_train = scaler[alg_id].transform(X_train)

            # get model event times and predicted survival functions for all training instances
            model = models[alg_id]
            event_times = model.event_times_
            survival_functions = model.predict_survival_function(X_train)

            # preprocess event_times and survival_functions
            event_times = np.append(0.0, event_times)
            event_times = np.append(event_times, self.algorithm_cutoff_time)             
            survival_functions = np.hstack([np.broadcast_to(1.0, shape=(np.size(survival_functions, axis=0), 1)), survival_functions])   

            # compute quantile and tail-value-at-risk for all alpha, for all instances
            for inst in range(np.size(X_train, axis=0)):
                for num, alpha in enumerate(alpha_domain):
                    # compute idx that marks the alpha-percentile
                    idx = survival_functions[inst, :] <= 1 - alpha
                    if np.any(idx):
                        idx = np.argmax(idx)
                        if not lower:
                            t_perf[alg_id, inst, num] = np.sum(survival_functions[inst, idx:] * np.diff(event_times[idx:]))
                            t_perf[alg_id, inst, num] += event_times[idx] * alpha
                        
                        else:
                            t_perf[alg_id, inst, num] = np.sum(survival_functions[inst, :idx] * np.diff(event_times[:idx + 1]))

                    else:
                        t_perf[alg_id, inst, num] = self.algorithm_cutoff_time    
                        
        selected = np.argmin(t_perf, axis=0)
        return np.take_along_axis(performances, selected, axis=1)


    def set_parameters(self, parametrization):
        self.n_estimators = parametrization["n_estimators"]
        self.min_samples_split = parametrization["min_samples_split"]
        self.min_samples_leaf = parametrization["min_samples_leaf"]
        self.min_weight_fraction_leaf = parametrization["min_weight_fraction_leaf"]
        self.max_features = parametrization["max_features"]
        self.bootstrap = parametrization["bootstrap"]
        self.oob_score = parametrization["oob_score"]