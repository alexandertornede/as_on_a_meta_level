import configparser
import pandas as pd
import matplotlib.pyplot as plt


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('conf/experiment_configuration.cfg'))
    return config



def generate_sbs_vbs_change_plots(small_scenarios:bool):
    dataframe = get_dataframe_for_sql_query("SELECT scenario_name, AVG(oracle_result_level_0) as VBS_0, AVG(oracle_result_level_1) as VBS_1, AVG(sbs_result_level_0) as SBS_0, AVG(sbs_result_level_1) as SBS_1 FROM `complete_sbs_vbs_and_gap_overview` GROUP BY scenario_name ORDER BY scenario_name")
    scenarios_with_large_numbers = ['PROTEUS-2014','QBF-2011','SAT03-16_INDU','SAT11-HAND','SAT11-INDU','SAT11-RAND', 'GLUHACK-18', 'SAT18-EXP']
    scenarios_to_remove = [x for x in dataframe['scenario_name'] if x not in scenarios_with_large_numbers] if not small_scenarios else [x for x in dataframe['scenario_name'] if x in scenarios_with_large_numbers]
    scenarios_to_remove.append('GRAPHS-2015')
    for scenario in scenarios_to_remove:
        dataframe = dataframe.drop(dataframe[dataframe['scenario_name'] == scenario].index)
    ax = dataframe.plot(kind='bar', x='scenario_name', figsize=(15,6), color=['#6ed95f','#29871c','#838dfc','#2230c9']) #RdYlBu
    ax.set_xlabel("Scenario")
    ax.set_ylabel("PAR10")
    ax.legend(['oracle', 'AS-oracle', 'SBS', 'SBAS'])
    vals = ax.get_yticks()
    for tick in vals:
        ax.axhline(y=tick, linestyle='dashed', alpha=0.4, color='#6d6e6d', zorder=1)

    name = 'plots/bars_small.pdf' if small_scenarios else 'plots/bars_large.pdf'
    plt.savefig(name, bbox_inches = "tight")
    print(dataframe.to_latex(index=False, float_format="%.3f"))

def generate_sbs_vbs_change_table():
    dataframe = get_dataframe_for_sql_query("SELECT scenario_name, AVG(oracle_result_level_0) as VBS_0, AVG(oracle_result_level_1) as VBS_1, AVG(oracle_level_1_div_oracle_level_0) as 'VBS_1/VBS_0', AVG(sbs_result_level_0) as SBS_0, AVG(sbs_result_level_1) as SBS_1, AVG(sbs_level_0_div_sbs_level_1) as 'SBS_0/SBS_1', AVG(sbs_result_level_0_div_oracle_result_level_0) as 'SBS/VBS 0', AVG(sbs_result_level_1_div_oracle_result_level_1) as 'SBS/VBS 1' FROM `complete_sbs_vbs_and_gap_overview` GROUP BY scenario_name ORDER BY scenario_name")
    print(dataframe.to_latex(index=False, float_format="%.3f"))


def generate_level_N_normalized_par10_table(level: int):
    dataframe = get_dataframe_for_sql_query("SELECT scenario_name, approach, AVG(n_par10) as avg_n_par10 FROM `normalized_par10_level_" + str(level) + "` WHERE approach != 'sbs' AND approach != 'sbs_with_feature_costs' AND approach != 'oracle' AND scenario_name != 'CSP-MZN-2013 ' GROUP BY scenario_name, approach ORDER BY scenario_name, avg_n_par10 ASC")
    dataframe = dataframe.pivot_table(values='avg_n_par10', index='scenario_name', columns='approach', aggfunc='first')
    dataframe = dataframe.rename(columns =  {'Expectation_algorithm_survival_forest' : 'R2SExp', 'PAR10_algorithm_survival_forest': 'R2SPAR10', 'isac' : 'ISAC', 'multiclass_algorithm_selector':'MLC', 'per_algorithm_RandomForestRegressor_regressor' : 'PAReg', 'satzilla-11' : 'SATzilla\'11', 'sunny': 'SUNNY'})
    print(dataframe.to_latex(float_format="%.2f", multirow=True)) #header=['R2SExp', 'R2SPAR10', 'ISAC', 'MLC', 'PAReg', 'SATzilla\'11', 'Sunny']

def generate_normalized_par10_table_normalized_by_level_0():
    dataframe = get_dataframe_for_sql_query("SELECT union_table.scenario_name, union_table.approach, AVG(union_table.n_par10) as avg_n_par10 FROM ((SELECT * FROM `normalized_by_level_0_par10_level_1` UNION SELECT * FROM `normalized_par10_level_0`) as union_table) WHERE approach NOT IN ('sbs', 'oracle', 'sbs_with_feature_costs','l1_sbs', 'l1_oracle', 'l1_sbs_with_feature_costs') AND scenario_name != 'CSP-MZN-2013 ' GROUP BY union_table.scenario_name, union_table.approach ORDER BY union_table.scenario_name, avg_n_par10")
    dataframe = dataframe.pivot_table(values='avg_n_par10', index='scenario_name', columns='approach', aggfunc='first')
    dataframe = dataframe.rename(
        columns={'Expectation_algorithm_survival_forest': 'R2SExp', 'PAR10_algorithm_survival_forest': 'R2SPAR10',
                 'isac': 'ISAC', 'multiclass_algorithm_selector': 'MLC',
                 'per_algorithm_RandomForestRegressor_regressor': 'PAReg', 'satzilla-11': 'SATzilla\'11',
                 'sunny': 'SUNNY', 'l1_Expectation_algorithm_survival_forest': 'L1_R2SExp', 'l1_PAR10_algorithm_survival_forest': 'L1_R2SPAR10',
                 'l1_isac': 'L1_ISAC', 'l1_multiclass_algorithm_selector': 'L1_MLC',
                 'l1_per_algorithm_RandomForestRegressor_regressor': 'L1_PAReg', 'l1_satzilla-11': 'L1_SATzilla\'11',
                 'l1_sunny': 'L1_SUNNY'})
    column_order = ['R2SExp','R2SPAR10','ISAC','MLC','SATzilla\'11','SUNNY','L1_R2SExp','L1_R2SPAR10','L1_ISAC','L1_MLC','L1_PAReg','L1_SATzilla\'11','L1_SUNNY']
    dataframe = dataframe[column_order]
    print(dataframe.to_latex(float_format="%.2f",
                             multirow=True))  # header=['R2SExp', 'R2SPAR10', 'ISAC', 'MLC', 'PAReg', 'SATzilla\'11', 'Sunny']

def get_dataframe_for_sql_query(sql_query: str ):
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

#generate_level_N_normalized_par10_table(level=1)
#generate_normalized_par10_table_normalized_by_level_0()
generate_sbs_vbs_change_plots(True)
generate_sbs_vbs_change_plots(False)