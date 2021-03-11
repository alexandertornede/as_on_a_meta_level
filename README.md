# as_on_a_meta_level
# Code for paper: "Algorithm Selection on a Meta Level"

This repository holds the code for our paper "Algorithm Selection on a Meta Level" by Alexander Tornede, Lukas Gehring, Tanja Tornede, Marcel Wever and Eyke Hüllermeier. Regarding questions please contact alexander.tornede@upb.de .

Please cite this work as
```
@inproceedings{tornede2020towards,
  title={Towards Meta-Algorithm Selection},
  author={Tornede, Alexander and Wever, Marcel and H{\"u}llermeier, Eyke},
  booktitle={Workshop on Meta-Learning (MetaLearn 2020) @ NeurIPS 2020},
  year={2020}
}
```

## Abstract
The problem of selecting an algorithm that appears most suitable for a specific instance of an algorithmic problem class, such as the Boolean satisfiability problem, is called instance-specific algorithm selection. Over the past decade, the problem has received considerable attention, resulting in a number of different methods for algorithm selection. Although most of these methods are based on machine learning, surprisingly little work has been done on meta learning, that is, on taking advantage of the complementarity of existing algorithm selection methods in order to combine them into a single superior algorithm selector. In this paper, we introduce the problem of meta algorithm selection, which essentially asks for the best way to combine a given set of algorithm selectors. We present a general methodological framework for meta algorithm selection as well as several concrete learning methods as instantiations of this framework, essentially combining ideas of meta learning and ensemble learning. In an extensive experimental evaluation, we demonstrate that ensembles of algorithm selectors can significantly outperform single algorithm selectors and have the potential to form the new state of the art in algorithm selection.

## Execution Details (Getting the Code to Run)
For the sake of reproducibility, we will detail how to reproduce the results presented in the paper below.

## 1. Configuration
In order to reproduce the results by running our code, we assume that you have a MySQL server with version >=5.7.9 running.

As a next step, you have to create a configuration file entitled `experiment_configuration.cfg` in the `conf` folder on the top level of your IDE project. This configuration file should contain the following information:

```
[DATABASE]
host = my.sqlserver.com
username = username
password = password
database = databasename
table = tablename
ssl = true

[EXPERIMENTS]
scenarios=ASP-POTASSCO,BNSL-2016,CPMP-2015,CSP-2010,CSP-MZN-2013,CSP-Minizinc-Time-2016,GLUHACK-2018,MAXSAT12-PMS,MAXSAT15-PMS-INDU,QBF-2011,SAT03-16_INDU,SAT12-INDU,SAT18-EXP
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
* `table`: the name of the table, where results should be stored. This is created automatically by the code if it does not exist yet and should NOT be created manually.
* `ssl`: whether ssl should be used or not

Entries below the `[EXPERIMENTS]` define which experiments will be run. The configuration above will produce the main results presented in the paper.

Note: Since the base learner predictions are precomputed for more efficient execution of the voting and stacking ensembles, the configuration must be executed with the `create_base_learner_prediction` approach before the voting and stacking approaches can be run.

## 2. Packages and Dependencies
For running the code several dependencies have to be fulfilled. The easiest way of getting there is by using [Anaconda](https://anaconda.org/). For this purpose you find an Anaconda environment definition called `meta_as.yml` in the `anaconda` folder at the top-level of this project.  Assuming that you have Anaconda installed, you can create an according environment with all required packages via

```
conda env create -f ensemble_environment.yml
``` 

which will create an environment named `ensemble_tests_env`. After it has been successfully installed, you can use 
```
conda activate ensemble_tests_env
```
to activate the environment and run the code (see step 4).

## 3. ASlib Data
Obviously, the code requires access to the ASLib scenarios in order to run the requested evaluations. It expects the ASLib scenarios (which can be downloaded from [Github](https://github.com/coseal/aslib_data)) to be located in a folder `data/aslib_data-master` on the top-level of your IDE project. I.e. your folder structure should look similar to this: 
```
./algorithm_selection_ensembles
./algorithm_selection_ensembles/pre_compute
./algorithm_selection_ensembles/ensembles
./algorithm_selection_ensembles/survival_approaches
./algorithm_selection_ensembles/singularity
./algorithm_selection_ensembles/baselines
./algorithm_selection_ensembles/data
./algorithm_selection_ensembles/data/aslib_data-master
./algorithm_selection_ensembles/conf
./algorithm_selection_ensembles/predictions
./algorithm_selection_ensembles/logs
./algorithm_selection_ensembles/weights
```


## 5. Obtaining Evaluation Results
At this point you should be good to go and can execute the experiments by running the `run.py` on the top-level of the project. Again, before executing the voting and stacking approaches, the `create_base_learner_prediction` approach must first be executed to precompute the predictions of the base learner.

 All results will be stored in the table given in the configuration file and has the following columns:

* `scenario_name`: The name of the scenario.
* `fold`: The train/test-fold associated with the scenario which is considered for this experiment
* `approach`: The approach which achieved the reported results, where `Run2SurviveExp := Expectation_algorithm_survival_forest`, `Run2SurvivaPar10 := PAR10_algorithm_survival_forest`
* `metric`: The metric which was used to generate the result. For the `number_unsolved_instances` metric, the suffix `True` indicates that feature costs are accounted for whereas for `False` this is not the case. All other metrics automatically incorporate feature costs.
* `result`: The output of the corresponding metric.

The results obtained this way are (among others) PAR10 scores (not yet normalized) for the meta-AS problem. 

After having generated the main results for the meta-level, you also have to generate the results for the base-level by running the code on the original ASlib scenarios. For doing so, change  the `table` name in the configuration file (in order to avoid an overwrite) to another value and the `data_folder` to the folder where you downloaded the original ASlib data to. Furthermore, in the `approaches` part of the configuration, you should replace the `sbs_with_feature_costs` simply by `sbs`. Now you have to rerun the experiments to obtain the same table structure as described above.

At this point, you should have both the results for the base-level (we will call the corresponding results table `server_results_meta_level_0`) and for the meta-level (we will call the corresponding result table `server_results_meta_level_1`). In the following, you have to perform some data wrangling in order to generate the results presented in the paper.

### SQL Table Preparation
In order to generate the tables and plots presented in the paper, we ask you to create the following views in your SQL table: 

* Name: `oracle_vbs_gap_level_0` </br>
  Query: 
  ````
  SELECT oracle_level_0.scenario_name, oracle_level_0.fold, oracle_result_level_0, sbs_result_level_0, sbs_result_level_0/oracle_result_level_0 FROM (SELECT scenario_name, fold, result as oracle_result_level_0 FROM `server_results_meta_level_0` WHERE approach="oracle" AND metric="par10") as oracle_level_0 JOIN (SELECT scenario_name, fold, result as sbs_result_level_0 FROM `server_results_meta_level_0` WHERE approach="sbs" AND metric="par10") as sbs_level_0 ON oracle_level_0.scenario_name = sbs_level_0.scenario_name AND oracle_level_0.fold = sbs_level_0.fold
  ````
* Name: `oracle_vbs_gap_level_1` </br>
  Query: 
  ````
  SELECT oracle_level_1.scenario_name, oracle_level_1.fold, oracle_result_level_1, sbs_result_level_1, sbs_result_level_1/oracle_result_level_1 FROM (SELECT scenario_name, fold, result as oracle_result_level_1 FROM `server_results_meta_level_1` WHERE approach="oracle" AND metric="par10") as oracle_level_1 JOIN (SELECT scenario_name, fold, result as sbs_result_level_1 FROM `server_results_meta_level_1` WHERE approach="sbs_with_feature_costs" AND metric="par10") as sbs_level_1 ON oracle_level_1.scenario_name = sbs_level_1.scenario_name AND oracle_level_1.fold = sbs_level_1.fold
  ````
* Name: `complete_sbs_vbs_and_gap_overview` </br>
  Query:
  ````
  SELECT oracle_vbs_gap_level_0.scenario_name, oracle_vbs_gap_level_0.fold, oracle_vbs_gap_level_0.oracle_result_level_0, oracle_vbs_gap_level_1.oracle_result_level_1, oracle_vbs_gap_level_1.oracle_result_level_1 / oracle_vbs_gap_level_0.oracle_result_level_0 as oracle_level_1_div_oracle_level_0, oracle_vbs_gap_level_0.sbs_result_level_0, oracle_vbs_gap_level_1.sbs_result_level_1, oracle_vbs_gap_level_0.sbs_result_level_0/oracle_vbs_gap_level_1.sbs_result_level_1 as sbs_level_0_div_sbs_level_1, oracle_vbs_gap_level_0.sbs_result_level_0_div_oracle_result_level_0, oracle_vbs_gap_level_1.sbs_result_level_1_div_oracle_result_level_1  FROM `oracle_vbs_gap_level_0` JOIN `oracle_vbs_gap_level_1` ON oracle_vbs_gap_level_0.scenario_name = oracle_vbs_gap_level_1.scenario_name AND oracle_vbs_gap_level_0.fold = oracle_vbs_gap_level_1.fold
  ````

### Generating Plots
With the database views created as described above, you can simply run the `plot_generation.py` at the top-level of the project in order to obtain Figure 2 from the paper. 

### Generating Tables
If you want to re-generate the tables presented in the paper, navigate to the java folder: 
````shell
cd meta_as_code/java/
````
There you will find a gradle wrapper file as well as files containing database dumps ``table-data.kvcol`` and ``table2-data.kvcol``. In order to generate LaTeX tables run

````shell
./gradlew generateTables
````

if you are running a Linux or Mac OS and 

````shell
.\gradlew generateTables
````
if you have a Windows system. Running the command will result in three output files:  ``base-table.tex``,``win-tie-loss-stats.tex``, and ``meta-table.tex`` corresponding to Tables 1,2, and 3 respectively.

In case you want to generate tables from an existing database, you need to configure the ``database.properties`` file with the connection details and set the config constant ``LOAD_FROM_DB`` (``src/main/java/TableGenerator.java``) from ``false`` to ``true``.


## 6. Generating Scenarios for a New Meta-level
This part will most likely not be interesting for you and is only intended for developers: 

### Start Experiments for Meta-level N
1. Copy results for meta-level N-1 from the "output" folder on the server into a new subdirectory in output named "level_{N-1}".
2. Change "data_folder" in "conf/experiment_configuration" to "data/level_N/"
2. Change "table" in "conf/experiment_configuration" to "server_results_meta_level_N"
3. Change level in "meta_aslib_preparation" to level N
4. Run "meta_aslib_preparation" for meta-level N and copy the data from "data/level_N" to the server into the same directory
5. Check tables in SQL server
