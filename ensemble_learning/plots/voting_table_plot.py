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

    #voting_full = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting.approach, vbs_sbs.metric, voting.result, ((voting.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting ON vbs_sbs.scenario_name = voting.scenario_name AND vbs_sbs.fold = voting.fold AND vbs_sbs.metric = voting.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') AND approach LIKE '%%1_2_3_4_5_6_7%%' GROUP BY scenario_name, approach")
    voting_weighting_full = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_weighting.approach, vbs_sbs.metric, voting_weighting.result, ((voting_weighting.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_weighting ON vbs_sbs.scenario_name = voting_weighting.scenario_name AND vbs_sbs.fold = voting_weighting.fold AND vbs_sbs.metric = voting_weighting.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') AND approach LIKE '%%1_2_3_4_5_6_7%%' GROUP BY scenario_name, approach")
    voting_ranking_full = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, voting_ranking.approach, vbs_sbs.metric, voting_ranking.result, ((voting_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN voting_ranking ON vbs_sbs.scenario_name = voting_ranking.scenario_name AND vbs_sbs.fold = voting_ranking.fold AND vbs_sbs.metric = voting_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') AND approach LIKE '%%1_2_3_4_5_6_7%%' GROUP BY scenario_name, approach")

    satzilla = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_satzilla.approach, vbs_sbs.metric, boosting_satzilla.result, ((boosting_satzilla.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_satzilla ON vbs_sbs.scenario_name = boosting_satzilla.scenario_name AND vbs_sbs.fold = boosting_satzilla.fold AND vbs_sbs.metric = boosting_satzilla.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_satzilla%%' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP')")

    isac = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_isac.approach, vbs_sbs.metric, boosting_isac.result, ((boosting_isac.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_isac ON vbs_sbs.scenario_name = boosting_isac.scenario_name AND vbs_sbs.fold = boosting_isac.fold AND vbs_sbs.metric = boosting_isac.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_isac%%' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP')")

    # sunny, satzilla, peralgo
    bagging = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging.approach, vbs_sbs.metric, bagging.result, ((bagging.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging ON vbs_sbs.scenario_name = bagging.scenario_name AND vbs_sbs.fold = bagging.fold AND vbs_sbs.metric = bagging.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') GROUP BY scenario_name, approach")
    bagging_weighting = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_weighting.approach, vbs_sbs.metric, bagging_weighting.result, ((bagging_weighting.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_weighting ON vbs_sbs.scenario_name = bagging_weighting.scenario_name AND vbs_sbs.fold = bagging_weighting.fold AND vbs_sbs.metric = bagging_weighting.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') GROUP BY scenario_name, approach")
    bagging_ranking = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, bagging_ranking.approach, vbs_sbs.metric, bagging_ranking.result, ((bagging_ranking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN bagging_ranking ON vbs_sbs.scenario_name = bagging_ranking.scenario_name AND vbs_sbs.fold = bagging_ranking.fold AND vbs_sbs.metric = bagging_ranking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') GROUP BY scenario_name, approach")

    base_learner = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, baselines.approach, vbs_sbs.metric, baselines.result, ((baselines.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN baselines ON vbs_sbs.scenario_name = baselines.scenario_name AND vbs_sbs.fold = baselines.fold AND vbs_sbs.metric = baselines.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%multiclass%%' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') GROUP BY scenario_name")

    baselines = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, baselines.approach, vbs_sbs.metric, baselines.result, ((baselines.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN baselines ON vbs_sbs.scenario_name = baselines.scenario_name AND vbs_sbs.fold = baselines.fold AND vbs_sbs.metric = baselines.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') GROUP BY scenario_name, approach")

    sunny = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, adaboostsamme_sunny.approach, vbs_sbs.metric, adaboostsamme_sunny.result, ((adaboostsamme_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN adaboostsamme_sunny ON vbs_sbs.scenario_name = adaboostsamme_sunny.scenario_name AND vbs_sbs.fold = adaboostsamme_sunny.fold AND vbs_sbs.metric = adaboostsamme_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP')")

    multiclass = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, adaboostsamme_mulitclass.approach, vbs_sbs.metric, adaboostsamme_mulitclass.result, ((adaboostsamme_mulitclass.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN adaboostsamme_mulitclass ON vbs_sbs.scenario_name = adaboostsamme_mulitclass.scenario_name AND vbs_sbs.fold = adaboostsamme_mulitclass.fold AND vbs_sbs.metric = adaboostsamme_mulitclass.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_multiclass_algorithm_selector%%' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP')")

    per_algo = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, adaboostsamme_per_algo_40_it.approach, vbs_sbs.metric, adaboostsamme_per_algo_40_it.result, ((adaboostsamme_per_algo_40_it.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN adaboostsamme_per_algo_40_it ON vbs_sbs.scenario_name = adaboostsamme_per_algo_40_it.scenario_name AND vbs_sbs.fold = adaboostsamme_per_algo_40_it.fold AND vbs_sbs.metric = adaboostsamme_per_algo_40_it.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_per_algorithm_regressor%%' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP')")


    stacking_feature_selection = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking_feature_selection.approach, vbs_sbs.metric, stacking_feature_selection.result, ((stacking_feature_selection.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking_feature_selection ON vbs_sbs.scenario_name = stacking_feature_selection.scenario_name AND vbs_sbs.fold = stacking_feature_selection.fold AND vbs_sbs.metric = stacking_feature_selection.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND NOT approach LIKE '%%RandomForest%%' GROUP BY scenario_name, approach")
    stacking = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, stacking.approach, vbs_sbs.metric, stacking.result, ((stacking.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN stacking ON vbs_sbs.scenario_name = stacking.scenario_name AND vbs_sbs.fold = stacking.fold AND vbs_sbs.metric = stacking.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%1_2_3_4_5_6_7%%' AND NOT scenario_name='CSP-Minizinc-Obj-2016' AND NOT approach LIKE '%%RandomForest%%' GROUP BY scenario_name, approach")


    scenario_names = ["ASP-POTASSCO", "BNSL-2016", "CPMP-2015", "CSP-2010", "CSP-MZN-2013", "CSP-Minizinc-Time-2016",
                      "GLUHACK-2018", "MAXSAT12-PMS", "MAXSAT15-PMS-INDU", "QBF-2011", "SAT03-16_INDU", "SAT12-INDU",
                      "SAT18-EXP"]
    dfs = [sunny, multiclass, per_algo]
    max_it = 20

    approach_name = ["sunny", "multiclass_algorithm_selector", "per_algorithm_regressor", "satzilla", "isac"]
    boosting_data = []
    for i, df in enumerate(dfs):
        print(approach_name[i])
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
            stacking[stacking['approach'] == 'stacking_1_2_3_4_5_6_7Expectation'],
            stacking[stacking['approach'] == 'stacking_1_2_3_4_5_6_7SATzilla-11'], stacking_feature_selection[
                stacking_feature_selection[
                    'approach'] == 'stacking_1_2_3_4_5_6_7Expectation_full_fullvariance_threshold'],
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

    #result.columns = ['scenario_name', 'Voting wmaj', '_', 'Voting borda', '_', '_', 'Bagging SUNNY', '_', '_',
    #                  "Bagging SATzilla'11", '_', '_', 'Bagging PerAlgo', '_', '_', 'Bagging wmaj SUNNY', '_', '_',
    #                  "Bagging wmaj SATzilla'11", '_', '_', 'Bagging wmaj PerAlgo', '_', '_', 'Bagging borda SUNNY',
    #                  '_', '_', "Bagging borda SATzilla'11", '_', '_', 'Bagging borda PerAlgo', '_', '_',
    #                  'Stacking R2S-Exp', '_', '_', "Stacking SATzilla'11", '_', '_', 'Stacking VT R2S-Exp', '_', '_',
    #                  "Stacking VT SATzilla'11", 'Boosting SUNNY', 'Boosting Multiclass', 'Boosting PerAlgo', '_', '_', 'R2S-Exp', '_', '_', 'R2S-PAR10', '_', '_', 'ISAC', '_', '_',
    #                  'Multiclass', '_', '_', 'PerAlgo', '_', '_', "SATzilla'11", '_', '_', 'SUNNY', '_']

    result.columns = ['scenario_name', 'Voting wmaj', '_', 'Voting borda', '_', '_', '_', '_', '_',
                      "_", '_', '_', '_', '_', '_', 'Bagging wmaj SUNNY', '_', '_',
                      "_", '_', '_', '_', '_', '_', '_',
                      '_', '_', "_", '_', '_', 'Bagging borda PerAlgo', '_', '_',
                      'Stacking R2S-Exp', '_', '_', "_", '_', '_', '_', '_', '_',
                      "Stacking VT SATzilla'11", '_', 'Boosting Multiclass', 'Boosting PerAlgo', '_', '_',
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
    result = result.round(2)
    print(result)
    print(result.to_latex(index=False, columns=["scenario_name", "Voting wmaj", "Voting borda", "Bagging wmaj SUNNY", "Bagging borda PerAlgo", "Stacking R2S-Exp", "Stacking VT SATzilla'11", "Boosting Multiclass", "Boosting PerAlgo", 'R2S-Exp', 'R2S-PAR10', 'ISAC', 'Multiclass', 'PerAlgo', "SATzilla'11", 'SUNNY']))
    #print(result.to_latex(index=False, columns=['scenario_name', 'Voting wmaj', 'Voting borda', 'Bagging SUNNY', "Bagging SATzilla'11", 'Bagging PerAlgo', 'Bagging wmaj SUNNY', "Bagging wmaj SATzilla'11", 'Bagging wmaj PerAlgo', 'Bagging borda SUNNY', "Bagging borda SATzilla'11", 'Bagging borda PerAlgo', 'Stacking R2S-Exp', "Stacking SATzilla'11", 'Stacking VT R2S-Exp', "Stacking VT SATzilla'11", 'Boosting SUNNY', 'Boosting Multiclass', 'Boosting PerAlgo', 'R2S-Exp', 'R2S-PAR10', 'ISAC', 'Multiclass', "SATzilla'11", "PerAlgo", "SUNNY"]))




    baselines_results_mean = []

    baselines_results_median = []


    #for app in baselines.approach.unique():
    #    baselines_results_mean.append(np.average(baselines.loc[baselines['approach'] == app].result))
    #    baselines_results_median.append(np.median(baselines.loc[baselines['approach'] == app].result))



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

