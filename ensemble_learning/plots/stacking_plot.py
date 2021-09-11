import pandas as pd
import configparser
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('../conf/experiment_configuration.cfg'))
    return config


def plot():
    color1 = '#264653'
    color2 = '#2a9d8f'
    color3 = '#e76f51'
    color4 = '#e9c46a'
    color5 = '#251314'

    stacking_feature_selection = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result, COUNT(n_par10) as counter FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP') AND approach LIKE 'stacking_1_2_3_4_5_6_7%%_full_fullvariance%' AND NOT approach LIKE '%RandomForest%' AND NOT approach LIKE '%SVM%' AND NOT approach LIKE '%Expectation%' GROUP BY approach ORDER BY approach")
    stacking = get_dataframe_for_sql_query(
        "SELECT approach, AVG(n_par10) as result, COUNT(n_par10) as counter FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE 'stacking_1_2_3_4_5_6_7%%_full_full' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP') AND NOT approach LIKE '%RandomForest%' AND NOT approach LIKE '%SVM%' AND NOT approach LIKE '%Expectation%' GROUP BY approach ORDER BY approach")
    #sat = get_dataframe_for_sql_query(
    #    "SELECT approach, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach='satzilla-11' AND NOT scenario_name='CSP-Minizinc-Obj-2016' GROUP BY approach")

    baselines = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP') AND (approach='Expectation_algorithm_survival_forest' OR approach='PAR10_algorithm_survival_forest' OR approach='isac' OR approach='multiclass_algorithm_selector' OR approach='per_algorithm_RandomForestRegressor_regressor' OR approach='satzilla-11' OR approach='sunny') GROUP BY scenario_name, approach")

    print(stacking)
    print(stacking_feature_selection)
    #print(stacking)
    #print(sat)
    #print(baseline)

    baselines_results_mean = []
    for app in baselines.approach.unique():
        baselines_results_mean.append(np.average(baselines.loc[baselines['approach'] == app].result))

    fig = plt.figure(1, figsize=(10, 5))

    ax = fig.add_subplot(111)

    #ax.axhline(float(sat.result), color=color1, linestyle='dashed', linewidth=2, zorder=10)
    #ax.text(5.9, float(sat.result) - 0.008, "SATzilla'11", ha='center', va='bottom', rotation=0, fontsize=7)

    width = 0.2  # the width of the bars
    ind = np.arange(len(stacking_feature_selection.result))
    index = 0
    permutation = [2, 4, 3, 1, 5, 0]
    for i in ind:
        print(permutation[i])
        print(stacking_feature_selection.result[i])
        ax.bar(permutation[i] - width, stacking_feature_selection.result[i], width, color=color3, zorder=6)
        ax.bar(permutation[i], stacking.result[i], width, color=color2, zorder=6)

    ax.bar(0 + width, baselines_results_mean[4], width, color=color1, zorder=6)
    ax.bar(1 + width, baselines_results_mean[6], width, color=color1, zorder=6)
    ax.bar(2 + width, baselines_results_mean[2], width, color=color1, zorder=6)
    ax.bar(3 + width, baselines_results_mean[5], width, color=color1, zorder=6)
    ax.bar(4 + width, baselines_results_mean[1], width, color=color1, zorder=6)
    ax.bar(5 + width, baselines_results_mean[3], width, color=color1, zorder=6)

    ax.set_xticks(range(len(permutation)))
    ax.set_xticklabels(ax.set_xticklabels(["PerAlgo", "SUNNY", "ISAC", "SATzilla'11", "R2S-PAR10", "Multiclass"]))


    #plt.xticks(rotation=90)

    ax.set_ylabel('nPAR10', fontsize=11)
    ax.set_xlabel('Meta-Learner', fontsize=11)

    #ax.set_ylim(bottom=0.3)
    #ax.set_ylim(top=0.7)

    plt.grid(b=True, which='major', linestyle='-', axis='y', zorder=0)

    #l1 = mpatches.Patch(color=color1, label="Stacking with SelectKBest")
    l2 = mpatches.Patch(color=color2, label="Stacking without feature selection")
    l3 = mpatches.Patch(color=color3, label="Stacking with VarianceThreshold")
    l4 = mpatches.Patch(color=color1, label="Base")

    plt.ylim(0.3, 0.95)
    plt.yticks(np.arange(0.3, 0.95, step=0.05))

    fig.legend(handles=[l3, l2, l4], loc=1, prop={'size': 13}, bbox_to_anchor=(0.88, 0.98))

    plt.show()

    fig.savefig("plotted/stacking_feature_selection.pdf", bbox_inches='tight')


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