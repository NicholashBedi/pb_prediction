# run from flash_results folder
#~/src/pb_prediction/flash_results$ python ../combine_flash_results.py
from os import listdir
import numpy as np
from pathlib import Path
from athletes_PBS import convert_to_string_from_seconds

personal_best = np.load('personal_best_2019.npy',allow_pickle='TRUE').item()


def print_list(list):
    for i in list:
        print(i)

def get_race_name(filename):
    filename_minus_path = filename.split("/")[1]
    underscore_split_filename = filename_minus_path.split("_")
    return underscore_split_filename[1] + " " + underscore_split_filename[2] + " 1500m Final"

def get_gender(filename):
    filename_minus_path = filename.split("/")[1]
    return filename_minus_path[0]

def get_output_line(entry, gender, race_name):
    tab_split_entry = entry.split("\t")
    if len(tab_split_entry) < 3:
        print(tab_split_entry)
    name = tab_split_entry[3]
    if name.lower() in personal_best.keys():
        athlete_pb = convert_to_string_from_seconds(personal_best[name.lower()])
        sb = athlete_pb
    else:
        athlete_pb = "UNKNOWN"
        sb = "UNKNOWN"
    place = tab_split_entry[0]
    m100 = "UNKNOWN"
    m200 = "UNKNOWN"
    split_init_index = 7
    if tab_split_entry[split_init_index] == '':
        split_init_index = 8
    m300 = tab_split_entry[split_init_index].split(" ")[0]
    m400 = "UNKNOWN"
    m500 = "UNKNOWN"
    m600 = "UNKNOWN"
    m700 = tab_split_entry[split_init_index + 1 ].split(" ")[0]
    m800 = "UNKNOWN"
    m900 ="UNKNOWN"
    m1000 ="UNKNOWN"
    m1100 = tab_split_entry[split_init_index + 2].split(" ")[0]
    m1200 = "UNKNOWN"
    m1300 = "UNKNOWN"
    m1400 = "UNKNOWN"
    m1500 = tab_split_entry[6]
    return name + "," + gender + "," + athlete_pb + "," + sb + "," \
         + race_name + "," + place + "," \
         + m100 + "," + m200 + "," + m300 + "," + m400 + "," + m500 + "," \
         + m600 + "," + m700 + "," + m800 + "," + m900 + "," + m1000 + "," \
         + m1100+ "," + m1200+ "," + m1300+ "," + m1400+ "," + m1500 +"\n"


path = Path(__file__).parent.absolute()

list_of_results = listdir(str(path/"flash_results_2019"))
list_of_results_path = ["flash_results_2019/" + result for result in list_of_results]


with open("combined_results_2019.txt", "w") as combined_results_file:
    combined_results_file.write("Name,Gender,PB,SB,Race,Place,100m,200m,300m,400m,500m,600m,700m,800m,900m,1000m,1100m,1200m,1300m,1400m,1500m\n")
    for result_name in list_of_results_path:
        if result_name != "flash_results_2019/combined_results.txt":
            print(result_name)
            gender = get_gender(result_name)
            race_name = get_race_name(result_name)
            with open(result_name, "r") as result_file:
                for entry in result_file:
                    if entry != '\n':
                        combined_results_file.write(get_output_line(entry,\
                                                                gender,\
                                                                race_name))
