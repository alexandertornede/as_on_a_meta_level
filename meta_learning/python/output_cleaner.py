from os import listdir
from os.path import join

input_folder = "output/level_0_old"
output_folder = "output/level_0"
word_to_exclude = "sbs_with_feature_costs"

for file_name in listdir(input_folder):
    with open(join(output_folder, file_name), "w") as output_file:
        with open(join(input_folder, file_name)) as input_file:
            for line in input_file.readlines():
                if not word_to_exclude in line:
                    output_file.write(line)
