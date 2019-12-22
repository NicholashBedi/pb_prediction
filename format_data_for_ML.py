import numpy as np
from athletes_PBS import convert_to_seconds_from_string

data_names = {'name':0,
                'is_women':1,
                'PB':2,
                'place':3,
                '300m':4,
                '700m':5,
                '1100m':6,
                '1500m':7}
input_names = {'is_women':0,
                'place':1,
                '300m':2,
                '700m':3,
                '1100m':4,
                '1500m':5}
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
    o_data["input"] = np.vstack((data[:,data_names['is_women']],
                            data[:,data_names['place']],
                            data[:,data_names['300m']],
                            data[:,data_names['700m']],
                            data[:,data_names['1100m']],
                            data[:,data_names['1500m']])).T
    o_data["target"] = data[:,data_names['PB']]
    o_data["name"] = data[:,data_names['name']]
    return o_data

def reformat_data(data):
    # Change gender to is_women
    replace_gender_string = {'w': 1, 'm': 0}
    data["input"][:, input_names['is_women']] = [replace_gender_string[gender] for gender in data["input"][:, input_names['is_women']]]

    # change times from string to number
    data["target"] = np.array([convert_to_seconds_from_string(string_time) for string_time in data["target"]], dtype=float)
    for idx in [input_names['300m'], input_names['700m'],
                input_names['1100m'], input_names['1500m']]:
        data["input"][:,idx] = [convert_to_seconds_from_string(string_time) for string_time in data["input"][:,idx]]
    data["input"] = data["input"].astype(float)
    data["target"] = data["input"][:,input_names['1500m']] - data["target"]
    # Change data to pace Difference
    distances_to_convert = [300, 700, 1100]
    for d in distances_to_convert:
        data["input"][:, input_names[str(d)+'m']] = \
                                  data["input"][:,input_names[str(d)+'m']]\
                                - (data["input"][:,input_names['1500m']]*(d/1500))
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
    data_type = ["input", "target", "name"]
    o_data["training"] = {}
    o_data["validation"] = {}
    o_data["testing"] = {}
    for type in data_type:
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

    data["norm_imput"] = np.linalg.norm(data["training"]["input"], axis=0)
    data["norm_imput"][data["norm_imput"] ==0] = 1
    data["norm_imput"][0] = 1
    for i in range(1,num_input): # Can skip 0 because it should be norm to 1
        data["training"]["input"][:,i] /= data["norm_imput"][i]
        data["validation"]["input"][:,i] /= data["norm_imput"][i]
        data["testing"]["input"][:,i] /= data["norm_imput"][i]

    data["mean_target"] = np.mean(data["training"]["target"])
    data["training"]["target"] -= data["mean_target"]
    data["validation"]["target"] -= data["mean_target"]
    data["testing"]["target"] -= data["mean_target"]

    data["norm_target"] = np.linalg.norm(data["training"]["target"], axis=0)
    data["training"]["target"] /= data["norm_target"]
    data["validation"]["target"] /= data["norm_target"]
    data["testing"]["target"] /= data["norm_target"]
    return data

def get_data(data_folder = "", data_name = "test_data",
            training_percent = 0.8, testing_percent = 0.1):
    data = np.genfromtxt(data_folder + data_name + ".txt",
                                  dtype=np.unicode_, delimiter=',', skip_header=1)
    data = discard_unknown_data(data)
    data = remove_unused_data(data)
    # New indexes - Can now use data_names dictionary
    # 0:Name,1:Gender,2:PB,3:Place,4:300m,5:700m,6:1100m,7:1500m
    np.random.seed(0)
    np.random.shuffle(data)
    data = break_into_name_input_target(data)
    data = reformat_data(data)

    data = break_data_into_components(data, training_percent, testing_percent)
    data = whiten_data(data)
    return data

def print_data(data):
    num_data = np.shape(data["name"])[0]
    for i in range(0, num_data):
        print(data["name"][i], end="; ")
        for j in range(0,6):
            print("{0:4.2f}".format(data["input"][i,j]), end="; ")
        print("\t", end="")
        print("{0:4.2f}".format(data["target"][i]))

if __name__ == '__main__':
    data_folder = ""
    data_name = "test_data"
    data = get_data(data_folder = data_folder, data_name = data_name,
                training_percent = 0.8, testing_percent = 0.1)

    print("\nTraining")
    print_data(data["training"])
    print("\nTesting")
    print_data(data["testing"])
    print("\nValidation")
    print_data(data["validation"])
