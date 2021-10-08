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

    # Test set
    #asp = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='ASP-POTASSCO'")
    #bnsl = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='BNSL-2016'")
    #cpmp = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='CPMP-2015'")
    #csp = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='CSP-2010'")
    #csp_time = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='CSP-Minizinc-Time-2016'")
    csp_mzn = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='CSP-MZN-2013'")
    #graphs = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='GRAPHS-2015'")
    #maxsat = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='MAXSAT-PMS-2016'")
    #maxsatwpms = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='MAXSAT-WPMS-2016'")
    #maxsat12 = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='MAXSAT12-PMS'")
    #maxsat15 = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='MAXSAT15-PMS-INDU'")
    #mip = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='MIP-2016'")
    #proteus = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='PROTEUS-2014'")
    qbf11 = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='QBF-2011'")
    #qbf14 = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='QBF-2014'")
    #qbf16 = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='QBF-2016'")
    #sat03 = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='SAT03-16_INDU'")
    #sat11hand = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='SAT11-HAND'")
    #sat11indu = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='SAT11-INDU'")
    #sat11rand = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='SAT11-RAND'")
    #sat12 = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='SAT12-ALL'")
    #sat12hand = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='SAT12-HAND'")
    #sat12indu = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='SAT12-INDU'")
    #sat12rand = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='SAT12-RAND'")
    #sat15 = get_dataframe_for_sql_query(
    #    "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_sunny.approach, vbs_sbs.metric, boosting_sunny.result, ((boosting_sunny.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_sunny ON vbs_sbs.scenario_name = boosting_sunny.scenario_name AND vbs_sbs.fold = boosting_sunny.fold AND vbs_sbs.metric = boosting_sunny.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_sunny%%' AND scenario_name='SAT15-INDU'")

    isac_csp_mzn = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_isac.approach, vbs_sbs.metric, boosting_isac.result, ((boosting_isac.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_isac ON vbs_sbs.scenario_name = boosting_isac.scenario_name AND vbs_sbs.fold = boosting_isac.fold AND vbs_sbs.metric = boosting_isac.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_isac%%' AND scenario_name='SAT12-INDU'")
    isac_qbf11 = get_dataframe_for_sql_query(
        "SELECT scenario_name, approach, fold, n_par10 as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, boosting_isac.approach, vbs_sbs.metric, boosting_isac.result, ((boosting_isac.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN boosting_isac ON vbs_sbs.scenario_name = boosting_isac.scenario_name AND vbs_sbs.fold = boosting_isac.fold AND vbs_sbs.metric = boosting_isac.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%SAMME_isac%%' AND scenario_name='ASP-POTASSCO'")

    base_learner = get_dataframe_for_sql_query(
        "SELECT scenario_name, AVG(n_par10) as result FROM (SELECT vbs_sbs.scenario_name, vbs_sbs.fold, baselines.approach, vbs_sbs.metric, baselines.result, ((baselines.result - vbs_sbs.oracle_result)/(vbs_sbs.sbs_result -vbs_sbs.oracle_result)) as n_par10,vbs_sbs.oracle_result, vbs_sbs.sbs_result FROM (SELECT oracle_table.scenario_name, oracle_table.fold, oracle_table.metric, oracle_result, sbs_result FROM (SELECT scenario_name, fold, approach, metric, result as oracle_result FROM `vbs_sbs` WHERE approach='oracle') as oracle_table JOIN (SELECT scenario_name, fold, approach, metric, result as sbs_result FROM `vbs_sbs` WHERE approach='sbs') as sbs_table ON oracle_table.scenario_name = sbs_table.scenario_name AND oracle_table.fold=sbs_table.fold AND oracle_table.metric = sbs_table.metric) as vbs_sbs JOIN baselines ON vbs_sbs.scenario_name = baselines.scenario_name AND vbs_sbs.fold = baselines.fold AND vbs_sbs.metric = baselines.metric WHERE vbs_sbs.metric='par10') as final WHERE metric='par10' AND approach LIKE '%%sunny%%' GROUP BY scenario_name")

    max_it = 20
    #dfs = [asp, bnsl, cpmp, csp, csp_mzn, csp_time, graphs, maxsat, maxsatwpms, maxsat12, maxsat15, mip, proteus, qbf11,
    #       qbf14, qbf16, sat03, sat11hand, sat11indu, sat11rand, sat12, sat12hand, sat12indu, sat12rand, sat15]

    dfs = [csp_mzn, qbf11]
    isac_dfs = [isac_csp_mzn, isac_qbf11]

    average_performance = np.zeros(max_it)

    fig = plt.figure(1, figsize=(10, 10))

    for i, df in enumerate(dfs):
        if df.empty:
            continue
        pos = i
        #ax1 = fig.add_subplot(2, 13, pos + 1)
        ax1 = fig.add_subplot(2, 2, pos + 1)

        #ax1.axhline(np.average(base_learner.result[i]), color=color1, linestyle='dashed', linewidth=1.4)
        #print(base_learner.scenario_name[i])

        # code by https://stackoverflow.com/questions/23493374/sort-dataframe-index-that-has-a-string-and-number
        # ----------------------
        df['indexNumber'] = [int(i.split('_')[-1]) for i in df.approach]
        df.sort_values(['indexNumber', 'fold'], ascending=[True, True], inplace=True)
        df.drop('indexNumber', 1, inplace=True)

        # ----------------------

        best_data = {}
        plot_data = []

        t_best_data = {}
        t_plot_data = []
        for iter in range(1, max_it + 1):
            data = []
            t_data = []

            # test plot
            for fold in range(1, 11):
                approach = 'SAMME_sunny_%d' % (iter)
                val = df.loc[(df['approach'] == approach) & (df['fold'] == fold)].result
                if len(val) == 1:
                    key = str(fold)
                    best_data[key] = val.iloc[0]
                    data.append(best_data[key])
                else:
                    data.append(best_data[str(fold)])
            plot_data.append(np.average(data))

            # training plot
            for fold in range(1, 11):
                approach = 'training_SAMME_sunny_%d' % (iter)
                val = df.loc[(df['approach'] == approach) & (df['fold'] == fold)].result
                if len(val) == 1:
                    key = str(fold)
                    t_best_data[key] = val.iloc[0]
                    t_data.append(t_best_data[key])
                else:
                    t_data.append(t_best_data[str(fold)])
            t_plot_data.append(np.average(t_data))

        ax1.plot(range(1, max_it + 1), plot_data)
        ax1.plot(range(1, max_it + 1), t_plot_data)
        average_performance = average_performance + plot_data

        # ax1.plot(range(1, len(sorted_data) + 1), sorted_data)
        plt.xlabel('Iterations')
        plt.ylabel('nPAR10')

        plt.xlim((1, max_it + 1))

        plt.title("SUNNY " + df.scenario_name[0])

    for i, df in enumerate(isac_dfs):
        if df.empty:
            continue
        pos = i
        #ax1 = fig.add_subplot(2, 13, pos + 1)
        ax1 = fig.add_subplot(2, 2, pos + 3)

        #ax1.axhline(np.average(base_learner.result[i]), color=color1, linestyle='dashed', linewidth=1.4)
        #print(base_learner.scenario_name[i])

        # code by https://stackoverflow.com/questions/23493374/sort-dataframe-index-that-has-a-string-and-number
        # ----------------------
        df['indexNumber'] = [int(i.split('_')[-1]) for i in df.approach]
        df.sort_values(['indexNumber', 'fold'], ascending=[True, True], inplace=True)
        df.drop('indexNumber', 1, inplace=True)

        # ----------------------

        best_data = {}
        plot_data = []

        t_best_data = {}
        t_plot_data = []
        for iter in range(1, max_it + 1):
            data = []
            t_data = []

            # test plot
            for fold in range(1, 11):
                approach = 'SAMME_isac_%d' % (iter)
                val = df.loc[(df['approach'] == approach) & (df['fold'] == fold)].result
                if len(val) == 1:
                    key = str(fold)
                    best_data[key] = val.iloc[0]
                    data.append(best_data[key])
                else:
                    data.append(best_data[str(fold)])
            plot_data.append(np.average(data))

            # training plot
            for fold in range(1, 11):
                approach = 'training_SAMME_isac_%d' % (iter)
                val = df.loc[(df['approach'] == approach) & (df['fold'] == fold)].result
                if len(val) == 1:
                    key = str(fold)
                    t_best_data[key] = val.iloc[0]
                    t_data.append(t_best_data[key])
                else:
                    t_data.append(t_best_data[str(fold)])
            t_plot_data.append(np.average(t_data))

        ax1.plot(range(1, max_it + 1), plot_data)
        ax1.plot(range(1, max_it + 1), t_plot_data)
        average_performance = average_performance + plot_data

        # ax1.plot(range(1, len(sorted_data) + 1), sorted_data)
        plt.xlabel('Iterations')
        plt.ylabel('nPAR10')

        plt.xlim((1, max_it + 1))

        plt.title("ISAC " + df.scenario_name[0])

    plt.show()

    average_performance = average_performance / 13

    print(average_performance.tolist())

    fig.savefig("plotted/samme_sunny.pdf", bbox_inches='tight')

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

