import tensorflow as tf
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from scipy.stats import norm
import numpy as np
import format_data_for_ML
import athletes_PBS
from sklearn.cluster import KMeans
from sklearn.datasets.samples_generator import make_blobs

# Plots four data sets
# Data is the dictionary of data
# x,y,z and are the names (string) to access said data
def plot_data(data,x,y,z,c):
    fig = plt.figure()
    ax = plt.axes(projection='3d')

    ax.set_xlabel(x)
    ax.set_ylabel(y)
    ax.set_zlabel(z)

    # Data for three-dimensional scattered points
    xdata = data[:,data_names[x]]
    ydata = data[:,data_names[y]]
    zdata = data[:,data_names[z]]
    cdata = data[:,data_names[c]]

    ax.scatter3D(xdata, ydata, zdata, c=cdata);
    plt.show()

# Creates a dictionary with each place having an array of data associated to it
def sort_data_by_place(data, data_names):
    place = {}
    for i in range(1, 19):
        place[str(i)] = np.array(([]))

    for i in range(0, np.shape(data)[0]):
        place[str(int(data[i,data_names["place"]]))] = \
                np.append(place[str(int(data[i,data_names["place"]]))],
                  data[i,data_names["target"]])
    return place

# Plots bar graph of place data
def plot_bar_graph(height_vals, num_of_data, y_lab = ""):
    show_to_place = 15
    place_vals = np.arange(1,show_to_place+1, 1)
    height_vals_to_show = height_vals[:show_to_place]

    plt.bar(x = place_vals - 1, height = height_vals_to_show)
    plt.xticks(place_vals - 1, place_vals)
    plt.xlabel("Place")
    plt.ylabel(y_lab)
    for i, v in enumerate(height_vals_to_show):
        plt.text(i, v, "{0:2.0f}".format(num_of_data[i]),
                color='blue', fontweight='bold')
    plt.show()


# Find what place has most PBs run
# Plots bar graph of percent of PBs for a given place
def place_pb_relation(data, data_names):
    place = sort_data_by_place(data, data_names)
    place_percent_pbs = np.zeros(18)
    num_of_data = np.zeros(18)
    for i in range(1,19):
        place_percent_pbs[i-1] = np.mean(place[str(i)] < 0.1)
        num_of_data[i-1] = np.shape(place[str(i)])[0]
    plot_bar_graph(place_percent_pbs, num_of_data, y_lab = "Number of PBs")


# Find how place impacts target
# Plots bar graph of target for place
def place_bar_graph(data, data_names):
    place = sort_data_by_place(data, data_names)
    place_mean_target = np.zeros(18)
    num_of_data = np.zeros(18)
    for i in range(1,19):
        place_mean_target[i-1] = np.mean(place[str(i)])
        num_of_data[i-1] = np.shape(place[str(i)])[0]
    plot_bar_graph(place_mean_target, num_of_data, y_lab = "Mean Difference From Season End PB")

# Determine number of PBs
# input just target data
# Outputs number of pbs and number of not pbs
def number_of_pbs(target_data, limit = 0.001):
    num_of_pbs = 0
    num_not_pbs = 0
    for data in target_data:
        if data < limit:
            num_of_pbs += 1
        else:
            num_not_pbs += 1
    print("Num of PBs: " + str(num_of_pbs) + " Num not PBs: " + str(num_not_pbs))
    return num_of_pbs, num_not_pbs


# This is a comaprison for machine learning algorithm
# Find the average people run slower than their year end PB. Use that to predict
# the error
def mean_as_prediction(data, data_names, print_vals = False):
    mean_target = np.mean(data[:, data_names["target"]])
    mean_absolute_error_using_mean_as_prediction = \
                np.mean(np.absolute(data[:, data_names["target"]] - mean_target))
    if print_vals:
        print("Mean target: {0:4.2f} | Mean error: {1:4.2f}".format(
            mean_target, mean_absolute_error_using_mean_as_prediction))
    return mean_target, mean_absolute_error_using_mean_as_prediction

# Find the k_means elbow
def k_means_get_elbow(data):
    # X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    # plt.scatter(X[:,0], X[:,1])
    wcss = []
    max_cluster = 21
    for i in range(1, max_cluster):
        kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        kmeans.fit(data)
        wcss.append(kmeans.inertia_)
    plt.plot(range(1, max_cluster), wcss)
    plt.title('Elbow Method')
    plt.xlabel('Number of clusters')
    plt.ylabel('WCSS')
    plt.show()

def print_data(data):
    num_data = np.shape(data)[0]
    print(" {0:>6}; {1:>6}; {2:>6}; {3:>6}; {4:>6}; {5:>6}; {6:>6}".format(
            "women?", "place", "300m", "700m", "1100m", "final", "PB diff"))
    for i in range(0, num_data):
        for j in range(0,7):
            print(" {0:6.2f}".format(data[i,j]), end=";")
        print()

# Run k-means on set of data
def k_means(data, num_clusters):
    # X, y = make_blobs(n_samples=300, centers=4, cluster_std=0.60, random_state=0)
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', max_iter=300, n_init=10, random_state=0)
    pred_y = kmeans.fit_predict(data)
    catagory_data = {}
    for cluster in range(0,num_clusters):
        print(kmeans.cluster_centers_[cluster, :])
        catagory_data[cluster] = data[pred_y == cluster, :]
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlabel("300m")
    ax.set_ylabel("700m")
    ax.set_zlabel("PB_pred")
    ax.scatter3D(data[:,0], data[:,1], data[:,3], c = pred_y);
    plt.show()
    return

# Makes a histogram for a single set of data (1d numpy array)
def histogram(data, bins_i = 'auto', xlabel = "", ylabel = "Density (%)"):
    plt.hist(data, bins=bins_i, density=True)
    plt.title("Histogram with 'auto' bins")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    mean, std = norm.fit(data)
    p = norm.pdf(x, mean, std)
    print( "Mean {0:>10}: {1:3.2f} | Std: {2:3.2f}".format(xlabel, mean, std))
    plt.plot(x, p, 'k', linewidth=2)
    plt.show()
    return


if __name__ == "__main__":
    data_folder = ""
    data_name = "test_data"
    data_name = "combined_results_2019"
    data_ml_format = format_data_for_ML.get_data(data_folder = data_folder,
                                                data_name = data_name,
                                                training_percent = 1,
                                                testing_percent = 0,
                                                whiten = False)
    names = data_ml_format["training"]["name"]
    # print(data_ml_format["training"]["target"])
    data = np.column_stack((data_ml_format["training"]["input"],
                    data_ml_format["training"]["target"]))
    data_names = {"is_women": 0, "place": 1, "300m": 2, "700m":3, "1100m":4,
                "1500m":5, "target":6}
    print("Number of data: ", end="")
    num_data = np.shape(data)[0]
    print(np.shape(data))
    just_split_data = np.hstack((data[:,2:5], data[:,6:]))


    # place_bar_graph(data,data_names)
    # plot_data(data, x = "place", y = "700m", z = 'target', c = 'target')
    # _ , _ = mean_as_prediction(data, data_names, print_vals = True)
    # k_means_get_elbow(just_split_data)
    # k_means(just_split_data, num_clusters = 4)
    # place_pb_relation(data, data_names)
    # place_bar_graph(data, data_names)
    # values = ["300m", "700m", "1100m", "1500m", "target"]
    # for v in values:
    #     histogram(data[:,data_names[v]], xlabel = v)
    number_of_pbs(data[:,data_names["target"]])
