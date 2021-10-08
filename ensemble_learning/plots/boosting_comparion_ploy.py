import sys

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

    color1 = '#264653'
    color2 = '#2a9d8f'
    color3 = '#e76f51'
    color4 = '#e9c46a'
    color5 = '#251314'

    isac = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_isac%%' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP')")

    baselines = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT approach_results.scenario_name, approach_results.fold, baselines.approach, approach_results.metric, baselines.result, ((baselines.result - approach_results.oracle_result)/(approach_results.sbs_result -approach_results.oracle_result)) as n_par10,approach_results.oracle_result, approach_results.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `approach_results` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `approach_results` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as approach_results JOIN baselines ON approach_results.scenario_name = baselines.scenario_name AND approach_results.fold = baselines.fold AND approach_results.metric = baselines.metric WHERE approach_results.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP') AND (approach='Expectation_algorithm_survival_forest' OR approach='PAR10_algorithm_survival_forest' OR approach='isac' OR approach='multiclass_algorithm_selector' OR approach='per_algorithm_RandomForestRegressor_regressor' OR approach='satzilla-11' OR approach='sunny') GROUP BY scenario_name, approach")
    baselines = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP') AND (approach='Expectation_algorithm_survival_forest' OR approach='PAR10_algorithm_survival_forest' OR approach='isac' OR approach='multiclass_algorithm_selector' OR approach='per_algorithm_RandomForestRegressor_regressor' OR approach='satzilla-11' OR approach='sunny') GROUP BY scenario_name, approach")

    sunny = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP')")

    multiclass = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_multiclass_algorithm_selector%%' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP')")

    per_algo = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_per_algorithm_regressor%%' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP')")

    satzilla = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_satzilla%%' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP')")

    baselines = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(n_par10) as result, COUNT(n_par10) as num FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, approach_results.approach, vbs_sbs.metric, approach_results.result, ((approach_results.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN approach_results ON vbs_sbs.scenario_name = approach_results.scenario_name AND vbs_sbs.fold = approach_results.fold AND vbs_sbs.metric = approach_results.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP') AND (approach='Expectation_algorithm_survival_forest' OR approach='PAR10_algorithm_survival_forest' OR approach='isac' OR approach='multiclass_algorithm_selector' OR approach='per_algorithm_RandomForestRegressor_regressor' OR approach='satzilla-11' OR approach='sunny') GROUP BY scenario_name, approach")

    baselines_results_mean = []
    baselines_results_median = []

    for app in baselines.approach.unique():
        baselines_results_mean.append(np.average(baselines.loc[baselines['approach'] == app].result))
        baselines_results_median.append(np.median(baselines.loc[baselines['approach'] == app].result))



    fig = plt.figure(1, figsize=(15, 8))

    plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.4, hspace=None)

    ax = fig.add_subplot(121)
    ax.grid(axis='y', linestyle='-', linewidth=1, zorder=0)
    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1, step=0.1))
    l1 = mpatches.Patch(color=color1, label="Boosting")
    l2 = mpatches.Patch(color=color2, label="Base")
    l3 = mpatches.Patch(color=color1, label="Boosting")
    l4 = mpatches.Patch(color=color2, label="Base")
    plt.title("Boosting Ensemble (Mean)")
    plt.xlabel("Base Algorithm Selector")
    plt.ylabel("nPAR10 over all scenarios")

    plt.legend(handles=[l1, l2], loc=1)

    ax2 = fig.add_subplot(122)
    ax2.grid(axis='y', linestyle='-', linewidth=1, zorder=0)

    plt.ylim(0, 1)
    plt.yticks(np.arange(0, 1, step=0.1))

    plt.title("Boosting Ensemble (Median)")
    plt.xlabel("Base Algorithm Selector")
    plt.ylabel("nPAR10 over all scenarios")

    plt.legend(handles=[l3, l4], loc=1)



    scenario_names = ['ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT-PMS-2016', 'MAXSAT-WPMS-2016', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'PROTEUS-2014', 'QBF-2011', 'QBF-2014', 'QBF-2016', 'SAT03-16_INDU', 'SAT11-HAND', 'SAT11-INDU', 'SAT11-RAND', 'SAT12-ALL', 'SAT12-HAND', 'SAT12-INDU', 'SAT12-RAND', 'SAT15-INDU', 'SAT18-EXP']
    dfs = [sunny, multiclass, per_algo, isac, satzilla]
    max_it = 20

    approach_name = ["sunny", "multiclass_algorithm_selector", "per_algorithm_regressor", "isac", "satzilla"]
    baseline_data = [baselines[baselines['approach'] == 'sunny'], baselines[baselines['approach'] == 'multiclass_algorithm_selector'], baselines[baselines['approach'] == 'per_algorithm_RandomForestRegressor_regressor'], baselines[baselines['approach'] == 'isac'], baselines[baselines['approach'] == 'satzilla-11'], baselines[baselines['approach'] == 'Expectation_algorithm_survival_forest'], baselines[baselines['approach'] == 'PAR10_algorithm_survival_forest']]

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

        ax.bar((i + 1) - 0.1, np.average(plot_data), width=0.2, color=color1, zorder=6)
        ax.bar((i + 1) + 0.1, np.average(baseline_data[i].result), width=0.2, color=color2, zorder=6)
        ax2.bar((i + 1) - 0.1, np.median(plot_data), width=0.2, color=color1, zorder=6)
        ax2.bar((i + 1) + 0.1, np.median(baseline_data[i].result), width=0.2, color=color2, zorder=6)

        print(baseline_data[-1])
        ax.axhline(np.average(baseline_data[-1].result), color='#264653', linestyle='dashed', linewidth=1.4, zorder=10)
        ax.text(5.5, np.average(baseline_data[-1].result) - 0.03, "R2S-PAR10", rotation=90)

        ax2.axhline(np.median(baseline_data[-1].result), color='#264653', linestyle='dashed', linewidth=1.4, zorder=10)
        ax2.text(5.5, np.median(baseline_data[-1].result) - 0.03, "R2S-PAR10", rotation=90)

        ax.set_xticks([1, 2, 3, 4, 5])
        ax.set_xticklabels(["SUNNY", "Mutliclass", "PerAlgo", "ISAC", "SATzilla'11"])
        ax2.set_xticks([1, 2, 3, 4, 5])
        ax2.set_xticklabels(["SUNNY", "Mutliclass", "PerAlgo", "ISAC", "SATzilla'11"])

        # ax1.plot(range(1, len(sorted_data) + 1), sorted_data)

    plt.show()

    fig.savefig("plotted/SAMME_comparion.pdf", bbox_inches='tight')

    extent = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    #fig.savefig("plotted/SAMME_comparion1.pdf", bbox_inches=extent)
    fig.savefig("plotted/SAMME_comparion1.pdf", bbox_inches=extent.expanded(1.1, 1.2))

    extent = ax2.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig("plotted/SAMME_comparion2.pdf", bbox_inches=extent.expanded(1.1, 1.2))


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

