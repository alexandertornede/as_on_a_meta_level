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

    s1 = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(result) as result FROM survival_analysis WHERE metric='par1' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') GROUP BY approach, scenario_name")
    s2 = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(result) as result FROM survival_analysis WHERE metric='par2' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') GROUP BY approach, scenario_name")
    s3 = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(result) as result FROM survival_analysis WHERE metric='par3' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') GROUP BY approach, scenario_name")
    s4 = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(result) as result FROM survival_analysis WHERE metric='par4' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') GROUP BY approach, scenario_name")
    s5 = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(result) as result FROM survival_analysis WHERE metric='par5' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') GROUP BY approach, scenario_name")
    s6 = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(result) as result FROM survival_analysis WHERE metric='par6' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') GROUP BY approach, scenario_name")
    s7 = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(result) as result FROM survival_analysis WHERE metric='par7' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') GROUP BY approach, scenario_name")
    s8 = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(result) as result FROM survival_analysis WHERE metric='par8' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') GROUP BY approach, scenario_name")
    s9 = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(result) as result FROM survival_analysis WHERE metric='par9' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') GROUP BY approach, scenario_name")
    s10 = get_dataframe_for_sql_query(
        "SELECT approach, scenario_name, AVG(result) as result FROM survival_analysis WHERE metric='par10' AND scenario_name IN ('ASP-POTASSCO', 'BNSL-2016', 'CPMP-2015', 'CSP-2010', 'CSP-MZN-2013', 'CSP-Minizinc-Time-2016', 'GLUHACK-18', 'MAXSAT12-PMS', 'MAXSAT15-PMS-INDU', 'QBF-2011', 'SAT03-16_INDU', 'SAT12-INDU', 'SAT18-EXP') GROUP BY approach, scenario_name")

    data = [s1, s2,s3,s4,s5,s6,s7,s8,s9,s10]
    for survival_analysis in data:
        scenario_names = survival_analysis['scenario_name'].unique()
        appraoches = survival_analysis['approach'].unique()
        result = pd.DataFrame(index=scenario_names, columns=appraoches)

        for approach in appraoches:
            result[approach] = list(survival_analysis[survival_analysis['approach'].str.match(approach)]['result'])


        scenario_names = ["ASP-POTASSCO", "BNSL-2016", "CPMP-2015", "CSP-2010", "CSP-MZN-2013", "CSP-Minizinc-Time-2016",
                          "GLUHACK-2018", "MAXSAT12-PMS", "MAXSAT15-PMS-INDU", "QBF-2011", "SAT03-16_INDU", "SAT12-INDU",
                          "SAT18-EXP"]
        scenario_names = ["ASP-POTASSCO", "BNSL-2016", "CPMP-2015", "CSP-2010", "CSP-MZN-2013", "CSP-Minizinc-Time-2016", "MAXSAT12-PMS", "MAXSAT15-PMS-INDU", "QBF-2011", "SAT03-16_INDU", "SAT12-INDU"]
        max_it = 20

        approach_name = ["sunny", "multiclass_algorithm_selector", "per_algorithm_regressor", "satzilla", "isac"]
        boosting_data = []


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

        columns = list(appraoches)
        #columns.insert(0, "scenario_name")

        table = result.to_latex(columns=columns)
        print(table)

    return
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

