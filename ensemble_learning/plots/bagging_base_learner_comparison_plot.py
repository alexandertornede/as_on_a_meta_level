import pandas as pd
import configparser
from matplotlib import pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import matplotlib.patches as mpatches

def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('../conf/experiment_configuration.cfg'))
    return config


def plot():

    bagging = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP') AND approach LIKE '%bagging_10%%' AND approach NOT LIKE '%bagging_10%%_weighting' AND approach NOT LIKE '%bagging_10%%_ranking' GROUP BY scenario_name, approach")
    bagging_weighting = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP') AND approach LIKE '%bagging_10%%_weighting' GROUP BY scenario_name, approach")
    bagging_ranking = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP') AND approach LIKE '%bagging_10%%_ranking' GROUP BY scenario_name, approach")
    baselines = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP') AND (approach='Expectation_algorithm_survival_forest' OR approach='PAR10_algorithm_survival_forest' OR approach='isac' OR approach='multiclass_algorithm_selector' OR approach='per_algorithm_RandomForestRegressor_regressor' OR approach='satzilla-11' OR approach='sunny') GROUP BY scenario_name, approach")

    bagging_results_mean = []
    bagging_weighting_results_mean = []
    bagging_ranking_results_mean = []
    baselines_results_mean = []

    bagging_results_median = []
    bagging_weighting_results_median = []
    bagging_ranking_results_median = []
    baselines_results_median = []

    colors = ['#264653', '#2a9d8f', '#e76f51']

    for app in bagging.approach.unique():
        bagging_results_mean.append(np.average(bagging.loc[bagging['approach'] == app].result))
        bagging_results_median.append(np.median(bagging.loc[bagging['approach'] == app].result))

    for app in bagging_weighting.approach.unique():
        bagging_weighting_results_mean.append(np.average(bagging_weighting.loc[bagging_weighting['approach'] == app].result))
        bagging_weighting_results_median.append(np.median(bagging_weighting.loc[bagging_weighting['approach'] == app].result))

    for app in bagging_ranking.approach.unique():
        bagging_ranking_results_mean.append(np.average(bagging_ranking.loc[bagging_ranking['approach'] == app].result))
        bagging_ranking_results_median.append(np.median(bagging_ranking.loc[bagging_ranking['approach'] == app].result))

    for app in baselines.approach.unique():
        baselines_results_mean.append(np.average(baselines.loc[baselines['approach'] == app].result))
        baselines_results_median.append(np.median(baselines.loc[baselines['approach'] == app].result))

    # mean plot
    data_to_plot = [bagging_results_mean, bagging_weighting_results_mean, bagging_ranking_results_mean]
    baseline_data = [baselines[baselines['approach'] == 'isac'],
                     baselines[baselines['approach'] == 'multiclass_algorithm_selector'],
                     baselines[baselines['approach'] == 'per_algorithm_RandomForestRegressor_regressor'],
                     baselines[baselines['approach'] == 'satzilla-11'], baselines[baselines['approach'] == 'sunny'], baselines[baselines['approach'] == 'Expectation_algorithm_survival_forest'], baselines[baselines['approach'] == 'PAR10_algorithm_survival_forest']]

    fig = plt.figure(1, figsize=(15, 6))

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)

    ax = fig.add_subplot(121)
    ax.grid(axis='y', linestyle='-', linewidth=1, zorder=0)
    plt.ylim(0.2, 0.95)
    plt.yticks(np.arange(0.2, 0.96, step=0.05))

    bar_width = 0.2
    for i, app in enumerate(data_to_plot):
        pos = bar_width * (i - 1) - bar_width / 2
        ax.bar(0 + pos, app[0], width=bar_width, color=colors[i], zorder=6)
        ax.bar(1 + pos, app[1], width=bar_width, color=colors[i], zorder=6)
        ax.bar(4 + pos, app[2], width=bar_width, color=colors[i], zorder=6)
        ax.bar(3 + pos, app[3], width=bar_width, color=colors[i], zorder=6)
        ax.bar(2 + pos, app[4], width=bar_width, color=colors[i], zorder=6)
    ax.bar(0 + bar_width / 2 + bar_width, np.average(baseline_data[0].result), width=bar_width, color='#e9c46a',
            zorder=6)
    ax.bar(1 + bar_width / 2 + bar_width, np.average(baseline_data[1].result), width=bar_width, color='#e9c46a',
            zorder=6)
    ax.bar(2 + bar_width / 2 + bar_width, np.average(baseline_data[2].result), width=bar_width, color='#e9c46a',
            zorder=6)
    ax.bar(3 + bar_width / 2 + bar_width, np.average(baseline_data[3].result), width=bar_width, color='#e9c46a',
            zorder=6)
    ax.bar(4 + bar_width / 2 + bar_width, np.average(baseline_data[4].result), width=bar_width, color='#e9c46a',
            zorder=6)

    ax.axhline(np.average(baseline_data[-1].result), color='#264653', linestyle='dashed', linewidth=1.4, zorder=10)
    ax.text(4.7, np.average(baseline_data[-1].result) - 0.03, "R2S-PAR10", rotation=90)


    ax.set_xticks([0, 1, 2, 3, 4])
    ax.set_xticklabels(["ISAC", "Multiclass", "SUNNY", "SATzilla'11", "PerAlgo"])

    #ax.set_ylim(bottom=0)
    #ax.set_ylim(top=4)

    plt.title("Bagging Ensemble (Mean)")
    plt.xlabel("Base Algorithm Selector")
    plt.ylabel("nPAR10 over all scenarios")

    l1 = mpatches.Patch(color=colors[0], label="maj")
    l2 = mpatches.Patch(color=colors[1], label="wmaj")
    l3 = mpatches.Patch(color=colors[2], label="borda")
    l4 = mpatches.Patch(color='#e9c46a', label="base")

    plt.legend(handles=[l1, l2, l3, l4], loc=1)

    # median plot
    data_to_plot = [bagging_results_median, bagging_weighting_results_median, bagging_ranking_results_median]

    ax2 = fig.add_subplot(122)
    ax2.grid(axis='y', linestyle='-', linewidth=1, zorder=0)
    plt.ylim(0.2, 0.95)
    plt.yticks(np.arange(0.2, 0.96, step=0.05))

    bar_width = 0.2
    for i, app in enumerate(data_to_plot):
        pos = bar_width * (i - 1) - bar_width / 2
        ax2.bar(0 + pos, app[0], width=bar_width, color=colors[i], zorder=6)
        ax2.bar(1 + pos, app[1], width=bar_width, color=colors[i], zorder=6)
        ax2.bar(4 + pos, app[2], width=bar_width, color=colors[i], zorder=6)
        ax2.bar(3 + pos, app[3], width=bar_width, color=colors[i], zorder=6)
        ax2.bar(2 + pos, app[4], width=bar_width, color=colors[i], zorder=6)
    ax2.bar(0 + bar_width / 2 + bar_width, np.median(baseline_data[0].result), width=bar_width, color='#e9c46a', zorder=6)
    ax2.bar(1 + bar_width / 2 + bar_width, np.median(baseline_data[1].result), width=bar_width, color='#e9c46a', zorder=6)
    ax2.bar(2 + bar_width / 2 + bar_width, np.median(baseline_data[2].result), width=bar_width, color='#e9c46a', zorder=6)
    ax2.bar(3 + bar_width / 2 + bar_width, np.median(baseline_data[3].result), width=bar_width, color='#e9c46a', zorder=6)
    ax2.bar(4 + bar_width / 2 + bar_width, np.median(baseline_data[4].result), width=bar_width, color='#e9c46a', zorder=6)

    ax2.axhline(np.median(baseline_data[-1].result), color='#264653', linestyle='dashed', linewidth=1.4, zorder=10)
    ax2.text(4.7, np.median(baseline_data[-1].result) - 0.03, "R2S-PAR10", rotation=90)

    ax2.set_xticks([0, 1, 2, 3, 4])
    ax2.set_xticklabels(["ISAC", "Multiclass", "SUNNY", "SATzilla'11", "PerAlgo"])

    plt.title("Bagging Ensemble (Median)")
    plt.xlabel("Base Algorithm Selector")
    plt.ylabel("nPAR10 over all scenarios")

    plt.legend(handles=[l1, l2, l3, l4], loc=1)

    plt.show()

    fig.savefig("plotted/bagging_base_learner_comparion.pdf", bbox_inches='tight')

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    # fig.savefig("plotted/SAMME_comparion1.pdf", bbox_inches=extent)
    fig.savefig("plotted/bagging_base_learner_comparion1.pdf", bbox_inches=extent.expanded(1.25, 1.2))

    extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("plotted/bagging_base_learner_comparion2.pdf", bbox_inches=extent.expanded(1.25, 1.2))

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

