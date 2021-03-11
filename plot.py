import pandas as pd
import configparser
from matplotlib import pyplot as plt


def load_configuration():
    config = configparser.ConfigParser()
    config.read_file(open('conf/experiment_configuration.cfg'))
    return config


def generate_sbs_vbs_change_table():
    dataframe = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, AVG(result) as result FROM `bagging_regression` WHERE scenario_name='CPMP-2015' GROUP BY scenario_name, approach")
    dataframe = dataframe.cumsum()
    plt.plot(dataframe.approach, dataframe.result)
    #plt.axis([0, 24, 0, 50])
    #plt.xticks(rotation=90)
    plt.show()
    print(dataframe)


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
generate_sbs_vbs_change_table()