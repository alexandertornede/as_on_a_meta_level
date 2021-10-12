# Reproducing the Ensemble Learning Results (Section 5)

In order to reproduce the ensemble learning results (Section 5), please follow the steps below.

## 1. Configuration
In order to reproduce the results by running our code, we assume that you have a MySQL server with version >=5.7.9 running.

As a next step, you have to create a configuration file entitled `experiment_configuration.cfg` in a `conf` folder in the `ensemble_learning` directory. This configuration file should contain the following information:

```
[DATABASE]
host = my.sqlserver.com
username = username
password = password
database = databasename
table = tablename
ssl = true

[EXPERIMENTS]
scenarios='ASP-POTASSCO,BNSL-2016,CPMP-2015,CSP-2010,CSP-MZN-2013,CSP-Minizinc-Time-2016,GLUHACK-18,MAXSAT-PMS-2016,MAXSAT-WPMS-2016,MAXSAT12-PMS,MAXSAT15-PMS-INDU,PROTEUS-2014,QBF-2011,QBF-2014,QBF-2016,SAT03-16_INDU,SAT11-HAND,SAT11-INDU,SAT11-RAND,SAT12-ALL,SAT12-HAND,SAT12-INDU,SAT12-RAND,SAT15-INDU,SAT18-EXP
approaches=sbs,oracle,base_learner,voting,voting_borda,voting_weighting,voting_optimize,bagging-base_learner,bagging_weighting-base_learner,bagging_borda-base_learner,samme,stacking_meta_learner,stacking_feature_selection
amount_of_training_scenario_instances=-1
amount_of_cpus=12
tune_hyperparameters=0
train_status=standard
```

You have to adapt all entries below the `[DATABASE]` tag according to your database server setup. The entries have the following meaning:
* `host`: the address of your database server
* `username`: the username the code can use to access the database
* `password`: the password the code can use to access the database
* `database`: the name of the database where you imported the tables
* `table`: the name of the table, where results should be stored. This is created automatically by the code if it does not exist yet and should NOT be created manually. To be able to plot the data later, you have to put the results of `sbs` and `oracle` into the table `sbs_vbs` and the results of all other approaches into the table `approach_results`.
* `ssl`: whether ssl should be used or not

Entries below the `[EXPERIMENTS]` define which experiments will be run. The configuration above will produce the main results presented in the paper.

Note: Since the base learner predictions are precomputed for more efficient execution of the voting and stacking ensembles, the configuration must be executed with the `create_base_learner_prediction` approach before the voting and stacking approaches can be run.

## 2. Packages and Dependencies
For running the code several dependencies have to be fulfilled. The easiest way of getting there is by using [Anaconda](https://anaconda.org/). For this purpose you find an Anaconda environment definition called `ensemble_environment.yml` in the `anaconda` folder in the `ensemble_learning` directory.  Assuming that you have Anaconda installed, you can create an according environment with all required packages via

```
conda env create -f ensemble_environment.yml
``` 

which will create an environment named `ensemble_tests_env`. After it has been successfully installed, you can use 
```
conda activate ensemble_tests_env
```
to activate the environment and run the code (see step 4).

## 3. ASlib Data
Obviously, the code requires access to the ASLib scenarios in order to run the requested evaluations. It expects the ASLib scenarios (which can be downloaded from [Github](https://github.com/coseal/aslib_data)) to be located in a folder `data/aslib_data-master` in the `ensemble_learning` directory. I.e. your folder structure should look similar to this: 
```
./as_on_a_meta_level/ensemble_learning/
./as_on_a_meta_level/ensemble_learning/ensembles
./as_on_a_meta_level/ensemble_learning/conf
./as_on_a_meta_level/ensemble_learning/survival_approaches
./as_on_a_meta_level/ensemble_learning/anaconda
./as_on_a_meta_level/ensemble_learning/baselines
./as_on_a_meta_level/ensemble_learning/plots
./as_on_a_meta_level/ensemble_learning/data
./as_on_a_meta_level/ensemble_learning/data/aslib_data-master
```


## 4. Obtaining Evaluation Results
At this point you should be good to go and can execute the experiments by running the `run.py` on the top-level of the project. Again, before executing the voting and stacking approaches, the `create_base_learner_prediction` approach must first be executed to precompute the predictions of the base learner.

 All results will be stored in the table given in the configuration file, which has the following columns:

* `scenario_name`: The name of the scenario.
* `fold`: The train/test-fold associated with the scenario which is considered for this experiment
* `approach`: The approach which achieved the reported results, where `Run2SurviveExp := Expectation_algorithm_survival_forest`, `Run2SurvivaPar10 := PAR10_algorithm_survival_forest`
* `metric`: The metric which was used to generate the result. For the `number_unsolved_instances` metric, the suffix `True` indicates that feature costs are accounted for whereas for `False` this is not the case. All other metrics automatically incorporate feature costs.
* `result`: The output of the corresponding metric.

The results obtained this way are (among others) PAR10 scores (not yet normalized) of several ensembling approaches for the meta-as problem. 

### Generating Plots
By running the `generate_all_plots.py` file at the top level of the `ensemble_learning` directory, you can generate all the plots from the paper. Note that you must use the table names presented in Section 1 to plot the data. Alternatively you can adjust the table names in the plot generation scripts.
