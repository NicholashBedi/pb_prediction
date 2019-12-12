import numpy as np

data_folder = ""
data_name = "test_data"

all_data = np.genfromtxt(data_folder + data_name + ".txt",
                              dtype=np.unicode_, delimiter=',', skip_header=1)

# Remove unused data
# 0:Name,1:Gender,2:PB,3:SB,4:Race,5:Place,
# 6:100m,7:200m,8:300m,9:400m,10:500m,11:600m,12:700m,13:800m,14:900m,15:1000m,
# 16:1100m,17:1200m,18:1300m,19:1400m,20:1500m
all_data[:,0] = np.char.add(np.char.add(all_data[:,0], "; "), all_data[:,4]) # Combine Name and race
all_data = np.hstack((all_data[:,:17], all_data[:,20:])) # Remove 1300-1400m
all_data = np.hstack((all_data[:,:13], all_data[:,16:])) # Remove 800-1100m
all_data = np.hstack((all_data[:,:9], all_data[:,12:])) # Remove 400-600m
all_data = np.hstack((all_data[:,:6], all_data[:,8:])) # Remove 100-200m
all_data = np.hstack((all_data[:,:3], all_data[:,5:])) # Remove SB and race
# New indexes
# 0:Name,1:Gender,2:PB,3:Place,4:300m,5:700m,6:1100m,7:1500m

# Change gender to is_women
replace_gender_string = {'w': 1, 'm': 0}
data_names = {'name':0,
                'is_women':1,
                'PB':2,
                'place':3,
                '300m':4,
                '700m':5,
                '1100m':6,
                '1500m':7}
all_data[:, data_names['is_women']] = [replace_gender_string[gender] for gender in all_data[:, data_names['is_women']]]

print(all_data)
