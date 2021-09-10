import pandas as pd
import configparser
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from functools import reduce

def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('../conf/experiment_configuration.cfg'))
    return config


def plot():

    voting_weighting_full = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'GRAPHS-2015', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'MIP-2016', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP') AND approach LIKE '%%voting_weighting_1_2_3_4_5_6_7%%' GROUP BY scenario_name, approach")
    voting_ranking_full = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'GRAPHS-2015', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'MIP-2016', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP') AND approach LIKE '%%voting_ranking_1_2_3_4_5_6_7%%' GROUP BY scenario_name, approach")

    # sunny, satzilla, peralgo
    bagging = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'GRAPHS-2015', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'MIP-2016', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP') AND approach LIKE '%%bagging%%' AND approach NOT LIKE '%%bagging%%ranking' AND approach NOT LIKE '%%bagging%%weighting' GROUP BY scenario_name, approach")
    bagging_weighting = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'GRAPHS-2015', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'MIP-2016', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP') AND approach LIKE '%%bagging%%weighting' GROUP BY scenario_name, approach")
    bagging_ranking = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'GRAPHS-2015', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'MIP-2016', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP') AND approach LIKE '%%bagging%%ranking' GROUP BY scenario_name, approach")

    baselines = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'GRAPHS-2015', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'MIP-2016', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP') AND (approach='Expectation_algorithm_survival_forest' OR approach='PAR10_algorithm_survival_forest' OR approach='isac' OR approach='multiclass_algorithm_selector' OR approach='per_algorithm_RandomForestRegressor_regressor' OR approach='satzilla-11' OR approach='sunny') GROUP BY scenario_name, approach")

    sunny = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE 'SAMME_sunny%%' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'GRAPHS-2015', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'MIP-2016', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP')")

    multiclass = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE 'SAMME_multiclass_algorithm_selector%%' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'GRAPHS-2015', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'MIP-2016', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP')")

    per_algo = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE 'SAMME_per_algorithm_regressor%%' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'GRAPHS-2015', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'MIP-2016', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP')")

    #satzilla = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_satzilla%%' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'GRAPHS-2015', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'MIP-2016', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP')")

    #isac = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_isac%%' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'GRAPHS-2015', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'MIP-2016', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP')")

    stacking_feature_selection = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%%stacking_1_2_3_4_5_6_7%%threshold' GROUP BY scenario_name, approach")
    stacking = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%1_2_3_4_5_6_7%%' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND approach LIKE '%%stacking_1_2_3_4_5_6_7%%' AND approach NOT LIKE '%%stacking_1_2_3_4_5_6_7%%threshold' GROUP BY scenario_name, approach")

    #print(voting_weighting_full)
    #print(voting_ranking_full)
    #print(bagging)
    #print(bagging_weighting)
    #print(bagging_ranking)
    #print(baselines)
    #print(sunny)
    #print(multiclass)
    #print(per_algo)
    #print(stacking_feature_selection)
    #print(stacking)
    scenario_names = ["ASP-POTASSCO", "BNSL-2016", "CPMP-2015", "CSP-2010", "CSP-MZN-2013", "CSP-Minizinc-Time-2016",
                      "GLUHACK-2018", "MAXSAT12-PMS", "MAXSAT15-PMS-INDU", "QBF-2011", "SAT03-16_INDU", "SAT12-INDU",
                      "SAT18-EXP"]
    scenario_names = ['ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'GRAPHS-2015', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'MIP-2016', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP']
    dfs = [sunny, multiclass, per_algo]
    max_it = 20

    approach_name = ["sunny", "multiclass_algorithm_selector", "per_algorithm_regressor", "satzilla", "isac"]
    boosting_data = []
    for i, df in enumerate(dfs):
        if df.empty:
            continue

        # code by https://stackoverflow.com/questions/23493374/sort-dataframe-index-that-has-a-string-and-number
        # ----------------------
        df['indexNumber'] = [int(i.split('_')[-1]) for i in df.approach]
        df.sort_values(['indexNumber', 'fold'], ascending=[True, True], inplace=True)
        df.drop('indexNumber', 1, inplace=True)

        # ----------------------

        best_data = {}
        plot_data = []

        for iter in range(1, max_it + 1):
            data = []
            for scenario_name in scenario_names:
                for fold in range(1, 11):
                    approach = 'SAMME_%s_%d' % (approach_name[i], iter)
                    val = df.loc[(df['approach'] == approach) & (df['fold'] == fold) & (
                                df['scenario_name'] == scenario_name)].result

                    if len(val) == 1:
                        key = str(fold)
                        best_data[key] = val.iloc[0]
                        data.append(best_data[key])
                    else:
                        data.append(best_data[str(fold)])
                if iter == max_it:
                    plot_data.append(np.average(data))

        boosting_data.append(plot_data)

    data = [voting_weighting_full, voting_ranking_full, bagging[bagging['approach'] == 'bagging_10_sunny'],
            bagging[bagging['approach'] == 'bagging_10_satzilla-11'],
            bagging[bagging['approach'] == 'bagging_10_per_algorithm_RandomForestRegressor_regressor'],
            bagging_weighting[bagging_weighting['approach'] == 'bagging_10_sunny_weighting'],
            bagging_weighting[bagging_weighting['approach'] == 'bagging_10_satzilla-11_weighting'],
            bagging_weighting[
                bagging_weighting['approach'] == 'bagging_10_per_algorithm_RandomForestRegressor_regressor_weighting'],
            bagging_ranking[bagging_ranking['approach'] == 'bagging_10_sunny_ranking'],
            bagging_ranking[bagging_ranking['approach'] == 'bagging_10_satzilla-11_ranking'],
            bagging_ranking[
                bagging_ranking['approach'] == 'bagging_10_per_algorithm_RandomForestRegressor_regressor_ranking'],
            stacking[stacking['approach'] == 'stacking_1_2_3_4_5_6_7Expectation_full_full'],
            #stacking[stacking['approach'] == 'stacking_1_2_3_4_5_6_7SATzilla-11_full_full'],
            #stacking_feature_selection[
            #    stacking_feature_selection[
            #        'approach'] == 'stacking_1_2_3_4_5_6_7Expectation_full_fullvariance_threshold'],
            stacking_feature_selection[stacking_feature_selection[
                                           'approach'] == 'stacking_1_2_3_4_5_6_7SATzilla-11_full_fullvariance_threshold']]


    result = reduce(lambda left, right: pd.merge(left, right, on=['scenario_name'],
                                                    how='inner'), data)

    result['Boosting SUNNY'] = boosting_data[0]
    result['Boosting Multiclass'] = boosting_data[1]
    result['Boosting PerAlgo'] = boosting_data[2]
    result['Boosting SUNNY'] = result['Boosting SUNNY']
    result['Boosting Multiclass'] = result['Boosting Multiclass']
    result['Boosting PerAlgo'] = result['Boosting PerAlgo']

    data = [result, baselines[baselines['approach'] == 'Expectation_algorithm_survival_forest'],
            baselines[baselines['approach'] == 'PAR10_algorithm_survival_forest'],
            baselines[baselines['approach'] == 'isac'],
            baselines[baselines['approach'] == 'multiclass_algorithm_selector'],
            baselines[baselines['approach'] == 'per_algorithm_RandomForestRegressor_regressor'],
            baselines[baselines['approach'] == 'satzilla-11'], baselines[baselines['approach'] == 'sunny']]
    result = reduce(lambda left, right: pd.merge(left, right, on=['scenario_name'],
                                                 how='inner'), data)

    #result.columns = ['scenario_name', 'Voting wmaj', '_', 'Voting borda', '_', '_', '_', '_', '_',
    #                  "_", '_', '_', '_', '_', '_', 'Bagging wmaj SUNNY', '_', '_',
    #                  "_", '_', '_', '_', '_', '_', '_',
    #                  '_', '_', "_", '_', '_', 'Bagging borda PerAlgo', '_', '_',
    #                  'Stacking R2S-Exp', '_', '_', "_", '_', '_', '_', '_', '_',
    #                  "Stacking VT SATzilla'11", '_', 'Boosting Multiclass', 'Boosting PerAlgo', '_', '_',
    #                  'R2S-Exp', '_', '_', 'R2S-PAR10', '_', '_', 'ISAC', '_', '_',
    #                  'Multiclass', '_', '_', 'PerAlgo', '_', '_', "SATzilla'11", '_', '_', 'SUNNY', '_']

    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        print(result)

    result.columns = ['scenario_name', 'Voting wmaj', '_', 'Voting borda', '_', '_', '_', '_', '_',
                      "_", '_', '_', '_', '_', '_', 'Bagging wmaj SUNNY', '_', '_',
                      "_", '_', '_', '_', '_', '_', '_',
                      '_', '_', "_", '_', '_', 'Bagging borda PerAlgo', '_', '_',
                      'Stacking R2S-Exp', '_', '_',
                      "Stacking VT SATzilla'11", '_', '_', 'Boosting Multiclass', 'Boosting PerAlgo', '_',
                      'R2S-Exp', '_', '_', 'R2S-PAR10', '_', '_', 'ISAC', '_', '_',
                      'Multiclass', '_', '_', 'PerAlgo', '_', '_', "SATzilla'11", '_', '_', 'SUNNY', '_']

    result = result.drop(['_'], axis=1)

    rank = result.rank(axis=1, method='min')
    rank = rank.mean()
    mean = result.mean()
    median = result.median()
    result.loc['mean'] = mean
    result.loc['median'] = median
    result.loc['avg rank'] = rank
    result.at['mean', 'scenario_name'] = 'Mean'
    result.at['median', 'scenario_name'] = 'Median'
    result.at['avg rank', 'scenario_name'] = 'Avg. Rank'
    result = result.round(2)

    table = result.to_latex(index=False, columns=["scenario_name", "Voting wmaj", "Voting borda", "Bagging wmaj SUNNY", "Bagging borda PerAlgo", "Stacking R2S-Exp", "Stacking VT SATzilla'11", "Boosting Multiclass", "Boosting PerAlgo", 'R2S-Exp', 'R2S-PAR10', 'ISAC', 'Multiclass', 'PerAlgo', "SATzilla'11", 'SUNNY'])
    #print(result.to_latex(index=False, columns=['scenario_name', 'Voting wmaj', 'Voting borda', 'Bagging SUNNY', "Bagging SATzilla'11", 'Bagging PerAlgo', 'Bagging wmaj SUNNY', "Bagging wmaj SATzilla'11", 'Bagging wmaj PerAlgo', 'Bagging borda SUNNY', "Bagging borda SATzilla'11", 'Bagging borda PerAlgo', 'Stacking R2S-Exp', "Stacking SATzilla'11", 'Stacking VT R2S-Exp', "Stacking VT SATzilla'11", 'Boosting SUNNY', 'Boosting Multiclass', 'Boosting PerAlgo', 'R2S-Exp', 'R2S-PAR10', 'ISAC', 'Multiclass', "SATzilla'11", "PerAlgo", "SUNNY"]))

    table = table.split("\midrule")[1]

    table = table.split(r"\\")

    # check if approach is better than baseline
    baseline_df = result[['R2S-Exp', 'R2S-PAR10', 'ISAC', 'Multiclass', 'PerAlgo', "SATzilla'11", 'SUNNY']]
    approach_df = result[["Voting wmaj", "Voting borda", "Bagging wmaj SUNNY", "Bagging borda PerAlgo", "Stacking R2S-Exp", "Stacking VT SATzilla'11", "Boosting Multiclass", "Boosting PerAlgo"]]
    for i, minimum in enumerate(baseline_df.min(axis=1)):
        new_line = table[i].split('&')[0]

        for a in approach_df.iloc[i]:
            if a < minimum:
                new_line = new_line + ' & $\overline{ ' + str(a) + ' }$'
            else:
                new_line = new_line + ' & ' + str(a)

        for b in baseline_df.iloc[i]:
            new_line = new_line + ' & ' + str(b)

        table[i] = new_line


    # find best value
    for i, minimum in enumerate(result.min(axis=1)):
        table[i] = table[i] + ' '
        table[i] = table[i].replace(str(minimum) + " ", r'\mathbf{ ' + str(minimum) + ' } ')
        table[i] = table[i].replace('& \mathbf{ ' + str(minimum) + ' }', r'& $\mathbf{ ' + str(minimum) + ' }$')

    table = r'\\'.join(table)

    pre_table = r"""\begin{tabular}{l | r r | r r | r r | r r || r r r r r r r }
         \toprule
         \textbf{Ensemble} &
            \multicolumn{2}{c|}{Voting} &
            \multicolumn{2}{c|}{Bagging} &
            \multicolumn{2}{c|}{Stacking} &
            \multicolumn{2}{c||}{Boosting} & \\ \midrule
         \textbf{Aggregation} &
            \rotatebox{90}{wmaj} & \rotatebox{90}{borda} &
            \rotatebox{90}{wmaj} & \rotatebox{90}{borda} &
            \rotatebox{90}{R2S-Exp} & \rotatebox{90}{SATzilla'11 (VT)} &
            \rotatebox{90}{wmaj} & \rotatebox{90}{wmaj} &\\ \midrule
         \textbf{Base selector} &
            \rotatebox{90}{all} & \rotatebox{90}{all} &
            \rotatebox{90}{SUNNY} & \rotatebox{90}{PerAlgo} &
            \rotatebox{90}{all} & \rotatebox{90}{all} &
            \rotatebox{90}{Multiclass} & \rotatebox{90}{PerAlgo} &
            \rotatebox{90}{R2S-Exp} &
            \rotatebox{90}{R2S-PAR10} &
            \rotatebox{90}{ISAC} &
            \rotatebox{90}{Multiclass} &
            \rotatebox{90}{PerAlgo} &
            \rotatebox{90}{SATzilla'11} &
            \rotatebox{90}{SUNNY}\\
        \midrule
        \textbf{Scenario} \\
        \midrule"""

    table = pre_table + table

    table = table.split('Mean')
    table = table[0] + '\midrule \midrule\n Mean' + table[1]
    table = ''.join(table)
    print(table)



def get_dataframe_for_sql_query(sql_query: str):
    db_credentials = get_database_credential_string()
    return pd.read_sql(sql_query, con=db_credentials)


def get_database_credential_string():
    config = load_configuration()
    db_config_section = config['DATABASE']
    db_host = db_config_section['host']
    db_username = db_config_section['username']
    db_password = db_config_section['password']
    db_database = db_config_section['database']
    return "mysql://" + db_username + ":" + db_password + "@" + db_host + "/" + db_database


config = load_configuration()
plot()

