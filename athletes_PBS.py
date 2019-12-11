import math
from os import listdir
import numpy as np
from pathlib import Path

def round(n, decimals=0):
    multiplier = 10 ** decimals
    return math.floor(n*multiplier + 0.5) / multiplier

def convert_to_seconds_from_string(string_time):
    time_split_on_colon = string_time.split(":")
    minutes = float(time_split_on_colon[0])
    if "@" in time_split_on_colon[1]:
        seconds = float(time_split_on_colon[1].split("@")[0])
    elif "(" in time_split_on_colon[1]:
        seconds = float(time_split_on_colon[1].split("(")[0])
    else:
        seconds = float(time_split_on_colon[1])

    return round(minutes*60 + seconds, 2)

def convert_to_string_from_seconds(time_number):
    minutes = math.floor(time_number/60)
    seconds = time_number - 60*minutes
    if seconds < 10:
        str_seconds = "0" + str(round(seconds,2))
    else:
        str_seconds = str(round(seconds,2))
    return str(minutes) + ":" + str_seconds

def first_then_second_name(name):
    split_name = name.split(", ")
    return(split_name[1] + " " + split_name[0])

def get_athlete_name_and_PB(entry):
    split_entry = entry.split("\t")
    name = first_then_second_name(split_entry[1])
    time_in_sec = convert_to_seconds_from_string(split_entry[4])
    return name.lower(), time_in_sec

def print_dict(dict):
    for x in dict:
        print (x + ":\t\t" + convert_to_string_from_seconds(dict[x]))

def print_list(list):
    for i in list:
        print(i)

if __name__ == "__main__":
    personal_best = {}

    path = Path(__file__).parent.absolute()
    list_of_regions_women = listdir(str(path/"PBs_women_2019"))
    list_of_regions_women_path = ["PBs_women_2019/" + region for region in list_of_regions_women]

    list_of_regions_men = listdir(str(path/"PBs_men_2019"))
    list_of_regions_men_path = ["PBs_men_2019/" + region for region in list_of_regions_men]

    list_of_regions = list_of_regions_women_path + list_of_regions_men_path
    # Get personal bests from each region
    for region_name in list_of_regions:
        with open(region_name, "r") as region_file:
            for entry in region_file:
                if entry != '\n':
                    name, time_in_sec = get_athlete_name_and_PB(entry)
                    if name not in personal_best.keys():
                        personal_best[name] = time_in_sec
                    elif personal_best[name] > time_in_sec:
                        personal_best[name] = time_in_sec

    # Get other athlete personal bests
    with open("other_athletes_pbs.txt","r") as pb_list_file:
        for entry in pb_list_file:
            if entry != '\n' and entry[0] != "#":
                split_entry = entry.split(";")
                name = split_entry[0].lower()
                time_in_sec = convert_to_seconds_from_string(split_entry[1])
                if name not in personal_best.keys():
                    personal_best[name] = time_in_sec
                elif personal_best[name] > time_in_sec:
                    personal_best[name] = time_in_sec


    # print_dict(personal_best)
    np.save("personal_best_2019.npy", personal_best)
