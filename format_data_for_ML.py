# See get data function
import numpy as np
from athletes_PBS import convert_to_seconds_from_string

def remove_unused_data(data):
    # Remove unused data
    # 0:Name,1:Gender,2:PB,3:SB,4:Race,5:Place,
    # 6:100m,7:200m,8:300m,9:400m,10:500m,11:600m,12:700m,13:800m,14:900m,15:1000m,
    # 16:1100m,17:1200m,18:1300m,19:1400m,20:1500m
    data[:,0] = np.char.add(np.char.add(data[:,0], "; "), data[:,4]) # Combine Name and race
    data = np.hstack((data[:,:17], data[:,20:])) # Remove 1300-1400m
    data = np.hstack((data[:,:13], data[:,16:])) # Remove 800-1100m
    data = np.hstack((data[:,:9], data[:,12:])) # Remove 400-600m
    data = np.hstack((data[:,:6], data[:,8:])) # Remove 100-200m
    data = np.hstack((data[:,:3], data[:,5:])) # Remove SB and race
    return data

def break_into_name_input_target(data):
    o_data = {}
    data_names = {'name':0,
                    'is_women':1,
                    'PB':2,
                    'place':3,
                    '300m':4,
                    '700m':5,
                    '1100m':6,
                    '1500m':7}
    o_data["input"] = np.vstack((data[:,data_names['is_women']],
                            data[:,data_names['place']],
                            data[:,data_names['300m']],
                            data[:,data_names['700m']],
                            data[:,data_names['1100m']],
                            data[:,data_names['1500m']])).T
    o_data["target"] = data[:,data_names['PB']]
    o_data["name"] = data[:,data_names['name']]
    return o_data

# Changes data so that instead of m and w it is 1 and 0
# Changes times from string to float
# Changes target to how much slower this race was compared to PB (ie. How much faster athlete can go)
# Changes 300m, 700m and 1100m time to how far off pace they were at a given point
def reformat_data(data):
    # Change gender to is_women
    replace_gender_string = {'w': 1, 'm': 0}
    input_names = {'is_women':0,
                    'place':1,
                    '300m':2,
                    '700m':3,
                    '1100m':4,
                    '1500m':5}
    data["input"][:, input_names['is_women']] = [replace_gender_string[gender] for gender in data["input"][:, input_names['is_women']]]

    # change times from string to number
    data["target"] = np.array([convert_to_seconds_from_string(string_time) for string_time in data["target"]], dtype=float)
    for idx in [input_names['300m'], input_names['700m'],
                input_names['1100m'], input_names['1500m']]:
        data["input"][:,idx] = [convert_to_seconds_from_string(string_time) for string_time in data["input"][:,idx]]
    data["input"] = data["input"].astype(float)
    data["target"] = np.minimum(data["target"], data["input"][:,input_names['1500m']])
    # Change target to how much faster PB is compared to actual time
    data["target"] = data["input"][:,input_names['1500m']] - data["target"]
    # Change data to pace Difference
    distances_to_convert = [300, 700, 1100]
    for d in distances_to_convert:
        data["input"][:, input_names[str(d)+'m']] = \
                                (data["input"][:,input_names['1500m']]*(d/1500)) \
                                - data["input"][:,input_names[str(d)+'m']]
    return data


def break_data_into_components(data, training_percent = 0.8, testing_percent = 0.1):
    o_data = {}
    if (training_percent + testing_percent > 1):
        validation_percent = 0
        training_percent = training_percent/(training_percent + testing_percent)
        testing_percent = testing_percent/(training_percent + testing_percent)
    else:
        validation_percent = 1 - training_percent - testing_percent
    num_data = np.shape(data["name"])[0]
    o_data["training"] = {}
    o_data["validation"] = {}
    o_data["testing"] = {}
    for type in ["input", "target", "name"]:
        o_data["training"][type] = data[type][:round(num_data*training_percent)]
        o_data["validation"][type] = data[type][round(num_data*training_percent):
                             round(num_data*(training_percent+validation_percent))]
        o_data["testing"][type] = data[type][round(num_data*(training_percent+validation_percent)):]
    return o_data


def discard_unknown_data(data) :
    rows_of_unknown_pb = np.where(np.char.equal(data[:,2], 'UNKNOWN'))
    data = np.delete(data, rows_of_unknown_pb, 0)
    return data


def whiten_data(data):
    data["mean_input"] = np.mean(data["training"]["input"], axis=0)
    data["mean_input"][0] = 0.5
    num_input = np.shape(data["training"]["input"])[1]
    for i in range(0,num_input):
        data["training"]["input"][:,i] -= data["mean_input"][i]
        data["validation"]["input"][:,i] -= data["mean_input"][i]
        data["testing"]["input"][:,i] -= data["mean_input"][i]

    data["norm_input"] = np.linalg.norm(data["training"]["input"], axis=0)
    data["norm_input"][data["norm_input"] ==0] = 1
    data["norm_input"][0] = 1
    for i in range(1,num_input): # Can skip 0 because it should be norm to 1
        data["training"]["input"][:,i] /= data["norm_input"][i]
        data["validation"]["input"][:,i] /= data["norm_input"][i]
        data["testing"]["input"][:,i] /= data["norm_input"][i]

    data["mean_target"] = np.mean(data["training"]["target"])
    data["training"]["target"] -= data["mean_target"]
    data["validation"]["target"] -= data["mean_target"]
    data["testing"]["target"] -= data["mean_target"]

    data["norm_target"] = np.linalg.norm(data["training"]["target"], axis=0)
    data["training"]["target"] /= data["norm_target"]
    data["validation"]["target"] /= data["norm_target"]
    data["testing"]["target"] /= data["norm_target"]
    return data

# Returns data in the following format
# data["name"]:
#       Returns the athlete name and race they ran
# Next the data is broken into three sections:
# training, validation and testing
# eg. data["training"]
# In each of these section the data is formated as follows
# Target:
#       How much faster the athlete can go. (the difference between this races time and their PB)
#       (positive float)
# Input:
#       is_women: 0 if man, 1 if women
#       place: What place they finished in the race (positive int)
#       How much faster they were at 300m compared to an even split race (Float)
#       How much faster they were at 700m compared to an even split race (Float)
#       How much faster they were at 1100m compared to an even split race (Float)
#       Their 1500m finishing time
# Note, the data may be whitened
def get_data(data_folder = "", data_name = "test_data",
            training_percent = 0.8, testing_percent = 0.1,
            whiten = True, seed = 0):
    data = np.genfromtxt(data_folder + data_name + ".txt",
                                  dtype=np.unicode_, delimiter=',', skip_header=1)
    data = discard_unknown_data(data)
    data = remove_unused_data(data)
    # New indexes - Can now use data_names dictionary
    # 0:Name,1:Gender,2:PB,3:Place,4:300m,5:700m,6:1100m,7:1500m
    np.random.seed(seed)
    np.random.shuffle(data)
    data = break_into_name_input_target(data)
    data = reformat_data(data)

    data = break_data_into_components(data, training_percent, testing_percent)
    if (whiten):
        return whiten_data(data)
    return data

def print_data(data, keep_whiten = False, norm_input = [], mean_input = [],
                norm_target = 1, mean_target = 0, predictions = []):
    num_data = np.shape(data["name"])[0]
    is_prediction = (len(predictions) == num_data)
    print("{0:>40}|  {1:>6}; {2:>6}; {3:>6}; {4:>6}; {5:>6}; {6:>6};| {7:>7}".format(
            "name", "women?", "place", "300m", "700m", "1100m", "final", "PB diff"), end="")
    if is_prediction:
        print("| {0:>7}| {1:>7}".format("Predict", "Abs Error"))
        print("{0:->115}".format(""))
    else:
        print("\n{0:->99}".format(""))
    total_abs_error = 0
    total_square_error = 0
    for i in range(0, num_data):
        print("{0:<40}".format(data["name"][i]), end="| ")
        for j in range(0,6):
            if (keep_whiten):
                print(" {0:6.2f}".format(data["input"][i,j]), end=";")
            else:
                print("{0:6.2f}".format(data["input"][i,j] \
                                        * norm_input[j] \
                                        + mean_input[j]), end="; ")
        real_target = data["target"][i]*norm_target + mean_target
        print("| {0:7.2f}".format(real_target), end="")
        if (is_prediction):
            abs_error = abs(predictions[i] - real_target)
            total_square_error += (predictions[i] - real_target)**2
            total_abs_error += abs_error
            print("| {0:7.2f}| {1:6.2f}".format(predictions[i],abs_error))
    if num_data == 0:
        return 0, 0
    return total_abs_error/num_data, total_square_error/num_data

if __name__ == '__main__':
    data_folder = ""
    data_name = "test_data"
    data_name = "combined_results_2019"
    data = get_data(data_folder = data_folder, data_name = data_name,
                training_percent = 0.8, testing_percent = 0.1,
                whiten = False)

    print("\nTraining")
    print_data(data["training"])
    print("\nTesting")
    print_data(data["testing"])
    print("\nValidation")
    print_data(data["validation"])
