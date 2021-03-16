import re
import os
import logging

from distutils.dir_util import copy_tree
from shutil import copyfile

logger = logging.getLogger("meta_aslib_preparation")
logger.addHandler(logging.StreamHandler())

META_BASE_SCENARIO_FOLDER = "data/meta_base"

ALGORITHM_RUNS_FILE = "algorithm_runs.arff"
DESCRIPTION_FILE = "description.txt"


def get_all_scneario_folders(path_to_scenarios: str):
    directory = path_to_scenarios
    scenario_folders = [os.path.join(directory, f) for f in os.listdir(directory) if
                       os.path.isdir(os.path.join(directory, f))]
    return scenario_folders


def delete_algorithm_run_files_in_scenario_folder(scenario_folder: str):
    scenario_folders = get_all_scneario_folders(scenario_folder)

    for folder in scenario_folders:
        algorithm_file_name = os.path.join(folder, ALGORITHM_RUNS_FILE)

        # delete algorithm runs file if it exists
        if os.path.exists(algorithm_file_name):
            logger.info("Removing algorithm run files in meta-scenario " + str(folder))
            os.remove(algorithm_file_name)

def adapt_description_file_in_new_scenarios_scenarios(meta_level):
    scenario_folders = get_all_scneario_folders("data/level_" + str(meta_level))

    algorithms = ["sbs", "per_algorithm_RandomForestRegressor_regressor", "multiclass_algorithm_selector",
                  "satzilla-11", "isac", "sunny", "Expectation_algorithm_survival_forest",
                  "PAR10_algorithm_survival_forest"]

    for folder in scenario_folders:
        logger.info("Replacing algorithms in meta-scenario " + str(folder))

        #adapt algorithm names in description file
        opened_file = open(os.path.join(folder, DESCRIPTION_FILE) , 'r')
        file_content = ''.join(opened_file.readlines())
        algorithm_info_part = "\n    configuration: ''\n" +"    deterministic: true\n"
        algorithm_string = ""
        for algorithm_name in algorithms:
            algorithm_string += "  " + algorithm_name+":" + algorithm_info_part

        regex = 'metainfo_algorithms:.*\s(([^\S\r\n])+.*\s)*'

        file_content = re.sub(regex, "metainfo_algorithms:\n" + algorithm_string, file_content)

        #adapt performance measures to runtime

        regex = 'performance_measures:.*\s((-*[^\S\r\n])+.*\s)*'

        file_content = re.sub(regex, "performance_measures:\n  - runtime\n", file_content)


        writable_file = open(os.path.join(folder, DESCRIPTION_FILE), 'w')
        writable_file.write(file_content)

def remove_oracle_from_algorithm_runs_in_scenario_folder(path_to_scenarios:str):

    scenario_folders = get_all_scneario_folders(path_to_scenarios)

    for folder in scenario_folders:

        algorithm_runs_file = folder + "/" + ALGORITHM_RUNS_FILE

        words_making_a_line_to_be_removed = ['oracle']

        with open(algorithm_runs_file) as oldfile, open(algorithm_runs_file+".new", 'w') as newfile:
            for line in oldfile:
                if not any(bad_word in line for bad_word in words_making_a_line_to_be_removed):
                    newfile.write(line)
        os.rename(algorithm_runs_file+".new", algorithm_runs_file)


def copy_output_files_from_previous_level_to_data_of_next_meta_level(meta_level:int):
    scenario_names = [os.path.basename(scenario_folder) for scenario_folder in get_all_scneario_folders("data/level_" + str(meta_level))]

    for scenario_name in scenario_names:
        output_file_of_previous_meta_level = "output/level_" + str(meta_level-1)+"/" + scenario_name + ".arff"
        copyfile(output_file_of_previous_meta_level, "data/level_" + str(meta_level) + "/" + scenario_name + "/algorithm_runs.arff")


def prepare_data_for_metalevel(meta_level:int):

    #make a copy of the base directory for the level
    new_meta_level_directory = "data/level_" + str(meta_level)
    copy_tree(META_BASE_SCENARIO_FOLDER, new_meta_level_directory)

    #delete algorithm run files from the data for this new meta_level
    delete_algorithm_run_files_in_scenario_folder(new_meta_level_directory)

    #copy output files from previous level to data of next meta_level
    copy_output_files_from_previous_level_to_data_of_next_meta_level(meta_level)

    #remove oracle lines from copied algorithm run files
    remove_oracle_from_algorithm_runs_in_scenario_folder("data/level_" + str(meta_level))

    adapt_description_file_in_new_scenarios_scenarios(meta_level)

prepare_data_for_metalevel(1)

