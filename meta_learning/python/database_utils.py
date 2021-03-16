import mysql.connector
import logging

logger = logging.getLogger("db_utils")
logger.addHandler(logging.StreamHandler())

def initialize_mysql_db_and_table_name_from_config(config):
    db_config_section = config['DATABASE']
    db_host = db_config_section['host']
    db_username = db_config_section['username']
    db_password = db_config_section['password']
    db_database = db_config_section['database']

    db = mysql.connector.connect(
        host=db_host,
        user=db_username,
        passwd=db_password,
        database=db_database,
        use_pure=True
    )

    db_table = config["DATABASE"]["table"]

    return db, db_table

def create_table_if_not_exists(db_handle, table_name:str):
    db_cursor = db_handle.cursor()
    db_cursor.execute("SHOW TABLES")
    result_set = db_cursor.fetchall()
    existing_tables = list()
    for row in result_set:
        existing_tables.append(row[0])

    if table_name not in existing_tables:
        logger.info("Creating table %s since it does not exist yet.", table_name)
        db_cursor.execute("CREATE TABLE " + table_name +  " (scenario_name VARCHAR(255) NOT NULL, fold int NOT NULL, approach VARCHAR(255) NOT NULL, metric VARCHAR(255) NOT NULL, result double NOT NULL)")
        logger.info("Successfully created table '%s'.", table_name)