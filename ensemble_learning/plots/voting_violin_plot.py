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

    voting = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') AND approach LIKE '%voting%' AND (approach NOT LIKE '%voting_r%' AND approach NOT LIKE '%voting_w%' AND approach NOT LIKE '%votingl%') GROUP BY scenario_name, approach")

    voting_weighting = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') AND approach LIKE '%voting_weighting%' AND approach NOT LIKE '%voting_weightingl%' GROUP BY scenario_name, approach")
    voting_ranking = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') AND approach LIKE '%voting_ranking%' AND approach NOT LIKE '%voting_rankingl%' GROUP BY scenario_name, approach")

    voting_full = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') AND approach='voting_1_2_3_4_5_6_7' GROUP BY scenario_name, approach")
    voting_weighting_full = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') AND approach='voting_weighting_1_2_3_4_5_6_7' GROUP BY scenario_name, approach")
    voting_ranking_full = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') AND approach='voting_ranking_1_2_3_4_5_6_7' GROUP BY scenario_name, approach")

    voting_opt = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') AND approach LIKE '%votingl%' GROUP BY scenario_name")
    voting_ranking_opt = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') AND approach LIKE '%voting_rankingl%' GROUP BY scenario_name")
    voting_weighting_opt = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') AND approach LIKE '%voting_weightingl%' GROUP BY scenario_name")

    baselines = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT approach_results.scenario_name, approach_results.fold, baselines.approach, approach_results.metric, baselines.result, ((baselines.result - approach_results.oracle_result)/(approach_results.sbs_result -approach_results.oracle_result)) as n_par10,approach_results.oracle_result, approach_results.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `approach_results` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `approach_results` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as approach_results JOIN baselines ON approach_results.scenario_name = baselines.scenario_name AND approach_results.fold = baselines.fold AND approach_results.metric = baselines.metric WHERE approach_results.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') AND (approach='Expectation_algorithm_survival_forest' OR approach='PAR10_algorithm_survival_forest' OR approach='isac' OR approach='multiclass_algorithm_selector' OR approach='per_algorithm_RandomForestRegressor_regressor' OR approach='satzilla-11' OR approach='sunny') GROUP BY scenario_name, approach")


    voting_results_mean = []
    voting_weighting_results_mean = []
    voting_ranking_results_mean = []
    baselines_results_mean = []

    voting_results_median = []
    voting_weighting_results_median = []
    voting_ranking_results_median = []
    baselines_results_median = []

    color1 = '#264653'
    color2 = '#042A2B'
    color3 = '#e76f51'

    for app in voting.approach.unique():
        voting_results_mean.append(np.average(voting.loc[voting['approach'] == app].result))
        voting_results_median.append(np.median(voting.loc[voting['approach'] == app].result))

    for app in voting_weighting.approach.unique():
        voting_weighting_results_mean.append(np.average(voting_weighting.loc[voting_weighting['approach'] == app].result))
        voting_weighting_results_median.append(np.median(voting_weighting.loc[voting_weighting['approach'] == app].result))

    for app in voting_ranking.approach.unique():
        voting_ranking_results_mean.append(np.average(voting_ranking.loc[voting_ranking['approach'] == app].result))
        voting_ranking_results_median.append(np.median(voting_ranking.loc[voting_ranking['approach'] == app].result))

    for app in baselines.approach.unique():
        baselines_results_mean.append(np.average(baselines.loc[baselines['approach'] == app].result))
        baselines_results_median.append(np.median(baselines.loc[baselines['approach'] == app].result))

    print(baselines.approach)

    # mean plot
    data_to_plot = [voting_results_mean, voting_weighting_results_mean, voting_ranking_results_mean]

    fig = plt.figure(1, figsize=(10, 6))
    plt.subplots_adjust(wspace=0.3)

    ax = fig.add_subplot(121)
    ax.grid(axis='y', linestyle='-', linewidth=1)

    ax.violinplot(data_to_plot, showmeans=True, widths=0.3)

    plt.axhline(baselines_results_mean[5], color=color2, linestyle='dashed', linewidth=1)
    ax.text(3.3, float(baselines_results_mean[5]), "SATzilla'11", ha='left', va='center', fontsize=8, rotation=90)

    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["maj", "wmaj", "borda"])

    # opt point
    ax.plot(1, np.average(voting_opt.result), 'o', color='black')
    ax.plot(2, np.average(voting_weighting_opt.result), 'o', color='black')
    ax.plot(3, np.average(voting_ranking_opt.result), 'o', color='black')

    # full point
    ax.plot(1, np.average(voting_full.result), 'o', color='red')
    ax.plot(2, np.average(voting_weighting_full.result), 'o', color='red')
    ax.plot(3, np.average(voting_ranking_full.result), 'o', color='red')

    plt.ylim(0.2, 0.8)
    plt.yticks(np.arange(0.2, 0.8, step=0.05))

    plt.title("Voting Ensemble (Mean)")
    plt.xlabel("Aggregation")
    plt.ylabel("nPAR10 over all scenarios")

    # median plot
    data_to_plot = [voting_results_median, voting_weighting_results_median, voting_ranking_results_median]

    ax2 = fig.add_subplot(122)
    ax2.grid(axis='y', linestyle='-', linewidth=1)

    ax2.violinplot(data_to_plot, showmeans=True, widths=0.3)

    plt.axhline(baselines_results_median[0], color=color2, linestyle='dashed', linewidth=1)
    ax2.text(3.3, float(baselines_results_median[0]), "R2S-Exp", ha='left', va='center', fontsize=8, rotation=90)

    ax2.set_xticks([1, 2, 3])
    ax2.set_xticklabels(["maj", "wmaj", "borda"])

    # opt point
    ax2.plot(1, np.median(voting_opt.result), 'o', color='black')
    ax2.plot(2, np.median(voting_weighting_opt.result), 'o', color='black')
    ax2.plot(3, np.median(voting_ranking_opt.result), 'o', color='black')

    # full point
    ax2.plot(1, np.median(voting_full.result), 'o', color='red')
    ax2.plot(2, np.median(voting_weighting_full.result), 'o', color='red')
    ax2.plot(3, np.median(voting_ranking_full.result), 'o', color='red')

    plt.ylim(0.2, 0.8)
    plt.yticks(np.arange(0.2, 0.8, step=0.05))

    plt.title("Voting Ensemble (Median)")
    plt.xlabel("Aggregation")
    plt.ylabel("nPAR10 over all scenarios")


    plt.show()

    fig.savefig("plotted/voting_violinplot.pdf", bbox_inches='tight')

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

