import sys

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
    voting_full = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(result) as result, COUNT(result) as num FROM performance_scenarios WHERE metric='performance' AND scenario_name IN ('OPENML-WEKA-2017', 'TTP-2016') AND approach LIKE '%%voting_1_2_3_4_7%%' GROUP BY scenario_name, approach")
    voting_weighting_full = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(result) as result, COUNT(result) as num FROM performance_scenarios WHERE metric='performance' AND scenario_name IN ('OPENML-WEKA-2017', 'TTP-2016') AND approach LIKE '%%voting_weighting_1_2_3_4_7%%' GROUP BY scenario_name, approach")
    voting_ranking_full = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(result) as result, COUNT(result) as num FROM performance_scenarios WHERE metric='performance' AND scenario_name IN ('OPENML-WEKA-2017', 'TTP-2016') AND approach LIKE '%%voting_ranking_1_2_3_4_7%%' GROUP BY scenario_name, approach")

    bagging_sunny_weighting = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(result) as result, COUNT(result) as num FROM performance_scenarios WHERE metric='performance' AND scenario_name IN ('OPENML-WEKA-2017', 'TTP-2016') AND approach='bagging_10_sunny_weighting' GROUP BY scenario_name, approach")
    bagging_peralgo_ranking = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(result) as result, COUNT(result) as num FROM performance_scenarios WHERE metric='performance' AND scenario_name IN ('OPENML-WEKA-2017', 'TTP-2016') AND approach='bagging_10_per_algorithm_RandomForestRegressor_regressor_ranking' GROUP BY scenario_name, approach")

    per_algorithm_regressor = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(result) as result, COUNT(result) as num FROM performance_scenarios WHERE metric='performance' AND scenario_name IN ('OPENML-WEKA-2017', 'TTP-2016') AND approach='per_algorithm_RandomForestRegressor_regressor' GROUP BY scenario_name, approach")
    sunny = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(result) as result, COUNT(result) as num FROM performance_scenarios WHERE metric='performance' AND scenario_name IN ('OPENML-WEKA-2017', 'TTP-2016') AND approach='sunny' GROUP BY scenario_name, approach")
    isac = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(result) as result, COUNT(result) as num FROM performance_scenarios WHERE metric='performance' AND scenario_name IN ('OPENML-WEKA-2017', 'TTP-2016') AND approach='isac' GROUP BY scenario_name, approach")
    satzilla = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(result) as result, COUNT(result) as num FROM performance_scenarios WHERE metric='performance' AND scenario_name IN ('OPENML-WEKA-2017', 'TTP-2016') AND approach='satzilla-11' GROUP BY scenario_name, approach")
    multiclass = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(result) as result, COUNT(result) as num FROM performance_scenarios WHERE metric='performance' AND scenario_name IN ('OPENML-WEKA-2017', 'TTP-2016') AND approach='multiclass_algorithm_selector' GROUP BY scenario_name, approach")

    boosting_multi_weighting = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(result) as result, COUNT(result) as num FROM performance_scenarios WHERE metric='performance' AND scenario_name IN ('OPENML-WEKA-2017', 'TTP-2016') AND approach='SAMME_multiclass_algorithm_selector_20' GROUP BY scenario_name, approach")
    boosting_peralgo_weighting = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(result) as result, COUNT(result) as num FROM performance_scenarios WHERE metric='performance' AND scenario_name IN ('OPENML-WEKA-2017', 'TTP-2016') AND approach='SAMME_per_algorithm_regressor_20' GROUP BY scenario_name, approach")

    stacking_feature_selection = get_dataframe_for_sql_query(
       "SELECT scenario_name, AVG(result) as result, COUNT(result) as num FROM performance_scenarios WHERE metric='performance' AND scenario_name IN ('OPENML-WEKA-2017', 'TTP-2016') AND approach='stacking_1_2_3_4_7SATzilla-11_full_fullvariance_threshold' GROUP BY scenario_name, approach")
    stacking = get_dataframe_for_sql_query(
       "SELECT scenario_name, AVG(result) as result, COUNT(result) as num FROM performance_scenarios WHERE metric='performance' AND scenario_name IN ('OPENML-WEKA-2017', 'TTP-2016') AND approach='stacking_1_2_3_4_7SUNNY_full_full' GROUP BY scenario_name, approach")

    sbs = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(result) as result, COUNT(result) as num FROM performance_scenarios WHERE metric='performance' AND scenario_name IN ('OPENML-WEKA-2017', 'TTP-2016') AND approach='sbs' GROUP BY scenario_name, approach")
    oracle = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(result) as result, COUNT(result) as num FROM performance_scenarios WHERE metric='performance' AND scenario_name IN ('OPENML-WEKA-2017', 'TTP-2016') AND approach='oracle' GROUP BY scenario_name, approach")


    data = [voting_weighting_full,
            voting_ranking_full, bagging_sunny_weighting, bagging_peralgo_ranking, stacking, stacking_feature_selection, boosting_multi_weighting, boosting_peralgo_weighting, per_algorithm_regressor, sunny, isac, satzilla, multiclass, sbs, oracle]

    result = reduce(lambda left, right: pd.merge(left, right, on=['scenario_name'],
                                                 how='inner'), data)

    result.columns = ['scenario_name', 'Voting wmaj', '_', 'Voting borda', '_', 'Bagging SUNNY', '_', 'Bagging PerAlgo', '_', 'Stacking', '_', 'Stacking VT', '_', 'Boosting Multi', '_', 'Boosting PerAlgo', '_', 'PerAlgo', '_', 'SUNNY', '_', 'ISAC', '_', "SATzilla'11", '_',
                      "Multiclass", '_', "sbs", '_', "oracle", '_', ]

    result = result.drop(['_'], axis=1)

    #rank = result.rank(axis=1, method='min')
    #rank = rank.mean()
    #mean = result.mean()
    #median = result.median()
    #result.loc['mean'] = mean
    #result.loc['median'] = median
    #result.loc['avg rank'] = rank
    #result.at['mean', 'scenario_name'] = 'Mean'
    #result.at['median', 'scenario_name'] = 'Median'
    #result.at['avg rank', 'scenario_name'] = 'Avg. Rank'
    result = result.round(2)



    print_latex_table(result)

def print_latex_table(result):
    table = result.to_latex(index=False)
    # print(result.to_latex(index=False, columns=['scenario_name', 'Voting wmaj', 'Voting borda', 'Bagging SUNNY', "Bagging SATzilla'11", 'Bagging PerAlgo', 'Bagging wmaj SUNNY', "Bagging wmaj SATzilla'11", 'Bagging wmaj PerAlgo', 'Bagging borda SUNNY', "Bagging borda SATzilla'11", 'Bagging borda PerAlgo', 'Stacking R2S-Exp', "Stacking SATzilla'11", 'Stacking VT R2S-Exp', "Stacking VT SATzilla'11", 'Boosting SUNNY', 'Boosting Multiclass', 'Boosting PerAlgo', 'R2S-Exp', 'R2S-PAR10', 'ISAC', 'Multiclass', "SATzilla'11", "PerAlgo", "SUNNY"]))

    table = table.split("\midrule")[1]

    table = table.split(r"\\")

    # check if approach is better than baseline
    baseline_df = result[['ISAC', 'Multiclass', 'PerAlgo', "SATzilla'11", 'SUNNY', 'sbs', 'oracle']]
    approach_df = result[
        ['Voting wmaj', 'Voting borda', 'Bagging SUNNY', 'Bagging PerAlgo', 'Stacking', 'Stacking VT', 'Boosting Multi', 'Boosting PerAlgo']]
    for i, minimum in enumerate(baseline_df.max(axis=1)):
        new_line = table[i].split('&')[0]

        for a in approach_df.iloc[i]:
            if a > minimum:
                new_line = new_line + ' & $\overline{ ' + str(a) + ' }$'
            else:
                new_line = new_line + ' & ' + str(a)

        for b in baseline_df.iloc[i]:
            new_line = new_line + ' & ' + str(b)

        table[i] = new_line

    # find best value
    for i, minimum in enumerate(result[['ISAC', 'Multiclass', 'PerAlgo', "SATzilla'11", 'SUNNY', 'sbs', 'Voting wmaj', 'Voting borda', 'Bagging SUNNY', 'Bagging PerAlgo', 'Stacking', 'Stacking VT', 'Boosting Multi', 'Boosting PerAlgo']].max(axis=1)):
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
                \rotatebox{90}{PerAlgo} &
                \rotatebox{90}{SUNNY} &
                \rotatebox{90}{ISAC} &
                \rotatebox{90}{SATzilla'11} &
                \rotatebox{90}{Multiclass} &
                \rotatebox{90}{sbs} &
                \rotatebox{90}{oracle}\\
            \midrule
            \textbf{Scenario} \\
            \midrule"""

    table = pre_table + table

    #table = table.split('Mean')
    #table = table[0] + '\midrule \midrule\n Mean' + table[1]
    #table = ''.join(table)
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
