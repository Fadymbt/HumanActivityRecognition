import statistics

from sklearn.feature_selection import VarianceThreshold, SelectFromModel

import sliding_window as sw
import read_dataset as rd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix
from sklearn.svm import SVC, LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# running = running.sample(frac = 1).reset_index(drop = True)


def get_accuracy(y, y_pred, algorithm_name):
    f1_macro = "f1 score macro " + algorithm_name + " = " + str(f1_score(y, y_pred, average = 'macro'))
    f1_micro = "f1 score micro " + algorithm_name + " = " + str(f1_score(y, y_pred, average = 'micro'))
    accuracy = "accuracy " + algorithm_name + " = " + str(accuracy_score(y, y_pred))
    print(f1_macro)
    print(f1_micro)
    print(accuracy)
    return f1_macro + "\n" + f1_micro + "\n" + accuracy


def get_function_duration(start_time):
    end = time.time()
    hours, rem = divmod(end - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds), "\n")


def generate_confusion_matrix(algorithm, x, y, algorithm_name, index, acc):
    plot_confusion_matrix(algorithm, x, y)
    plt.title(algorithm_name + str(index + 1))
    plt.xticks(rotation = 90)
    plt.xlabel("Predicted label\n" + acc)
    plt.tight_layout()
    if not os.path.exists("Output/Subject" + "_" + str(index + 1)):
        os.makedirs("Output/Subject" + "_" + str(index + 1))
    plt.savefig("Output/Subject" + "_" + str(index + 1) + "/" + algorithm_name + str(index + 1) + ".jpg")
    plt.show()
    return plt


def do_windows_slicing(activity_windows):
    y = []
    x = []
    for windows in activity_windows:
        window = pd.DataFrame(windows)
        y.append(window.iloc[:, window.shape[1] - 1].value_counts().idxmax())
        window.drop([window.shape[1] - 1], axis = 1, inplace = True)
        window.reset_index(drop = True, inplace = True)
        numpy_window = window.to_numpy()
        x.append(numpy_window)
    return x, y


def calculate_features(x, y):
    for i in range(len(x)):
        features = pd.DataFrame(x[i])
        total_x_out = []
        for features_index in range(features.shape[1]):
            temp_calculate_features_fourier = np.absolute(np.fft.fft(features[features_index].to_numpy()))
            temp_calculate_features_mean = statistics.mean(features[features_index].to_numpy())
            temp_calculate_features_std = statistics.stdev(features[features_index].to_numpy())
            x_out = np.insert(temp_calculate_features_fourier[:int(len(temp_calculate_features_fourier) * 0.1)], 0, [temp_calculate_features_mean])
            x_out = np.insert(x_out, 0, [temp_calculate_features_std])
            total_x_out.append(x_out)
        total_x_out = np.concatenate(total_x_out)
        x[i] = total_x_out
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    pd_x_y = pd.concat([x, y], axis = 1)
    pd_x_y.dropna(inplace = True)
    new_x = pd_x_y.iloc[:, 0:pd_x_y.shape[1] - 1]
    new_y = pd_x_y.iloc[:, pd_x_y.shape[1] - 1]
    new_x = StandardScaler().fit_transform(new_x)
    # print(new_x)
    return new_x, new_y


start = time.time()
path = "Dataset"
dataset = rd.get_all_subjects_data(path)
print("Data extracted")
get_function_duration(start)

all_subjects_time = time.time()
for i in range(15):
    if i == 1:
        continue
    single_subject_time = time.time()
    # Running Dataframe from dataset
    running_total = []
    running_test = []
    for j in range(15):
        if j == 1:
            continue
        running_total_per_row = []
        # Acc forearm
        temp = dataset[j][1][1][0][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        running_total_per_row.append(pd.DataFrame(temp))
        # Acc waist
        temp = dataset[j][1][1][6][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        running_total_per_row.append(pd.DataFrame(temp))
        # Acc thigh
        temp = dataset[j][1][1][3][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        running_total_per_row.append(pd.DataFrame(temp))
        # Acc head
        temp = dataset[j][1][1][2][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        running_total_per_row.append(pd.DataFrame(temp))

        # Gyr forearm
        temp = dataset[j][1][2][1][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        running_total_per_row.append(pd.DataFrame(temp))
        # Gyr waist
        temp = dataset[j][1][2][2][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        running_total_per_row.append(pd.DataFrame(temp))
        # Gyr thigh
        temp = dataset[j][1][2][5][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        running_total_per_row.append(pd.DataFrame(temp))
        # Gyr head
        temp = dataset[j][1][2][6][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        running_total_per_row.append(pd.DataFrame(temp))

        # Mag forearm
        temp = dataset[j][1][3][1][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        running_total_per_row.append(pd.DataFrame(temp))
        # Mag waist
        temp = dataset[j][1][3][6][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        running_total_per_row.append(pd.DataFrame(temp))
        # Mag thigh
        temp = dataset[j][1][3][4][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        running_total_per_row.append(pd.DataFrame(temp))
        # Mag head
        temp = dataset[j][1][3][2][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        running_total_per_row.append(pd.DataFrame(temp))

        if i == j:
            running_test = pd.concat(running_total_per_row, axis = 1, ignore_index = True)
        else:
            running_total.append(pd.concat(running_total_per_row, axis = 1, ignore_index = True))
    running = pd.concat(running_total, axis = 0, ignore_index = True)
    running.dropna(inplace = True)
    running_test.dropna(inplace = True)
    running["label"] = "running"
    running_test["label"] = "running"
    # print(running.shape)

    # Sitting Dataframe from dataset
    sitting_total = []
    sitting_test = []
    for j in range(15):
        if j == 1:
            continue
        sitting_total_per_row = []
        # Acc forearm
        temp = dataset[j][2][1][5][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        sitting_total_per_row.append(pd.DataFrame(temp))
        # Acc waist
        temp = dataset[j][2][1][0][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        sitting_total_per_row.append(pd.DataFrame(temp))
        # Acc thigh
        temp = dataset[j][2][1][3][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        sitting_total_per_row.append(pd.DataFrame(temp))
        # Acc head
        temp = dataset[j][2][1][2][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        sitting_total_per_row.append(pd.DataFrame(temp))

        # Gyr forearm
        temp = dataset[j][2][2][2][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        sitting_total_per_row.append(pd.DataFrame(temp))
        # Gyr waist
        temp = dataset[j][2][2][5][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        sitting_total_per_row.append(pd.DataFrame(temp))
        # Gyr thigh
        temp = dataset[j][2][2][1][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        sitting_total_per_row.append(pd.DataFrame(temp))
        # Gyr head
        temp = dataset[j][2][2][3][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        sitting_total_per_row.append(pd.DataFrame(temp))

        # Mag forearm
        temp = dataset[j][2][3][1][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        sitting_total_per_row.append(pd.DataFrame(temp))
        # Mag waist
        temp = dataset[j][2][3][6][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        sitting_total_per_row.append(pd.DataFrame(temp))
        # Mag thigh
        temp = dataset[j][2][3][4][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        sitting_total_per_row.append(pd.DataFrame(temp))
        # Mag head
        temp = dataset[j][2][3][2][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        sitting_total_per_row.append(pd.DataFrame(temp))

        if i == j:
            sitting_test = pd.concat(sitting_total_per_row, axis = 1, ignore_index = True)
        else:
            sitting_total.append(pd.concat(sitting_total_per_row, axis = 1, ignore_index = True))
    sitting = pd.concat(sitting_total, axis = 0, ignore_index = True)
    sitting.dropna(inplace = True)
    sitting_test.dropna(inplace = True)
    sitting["label"] = "sitting"
    sitting_test["label"] = "sitting"
    # print(sitting.shape)

    # Standing Dataframe from dataset
    standing_total = []
    standing_test = []
    for j in range(15):
        if j == 1:
            continue
        standing_total_per_row = []
        # Acc forearm
        temp = dataset[j][3][1][0][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        standing_total_per_row.append(pd.DataFrame(temp))
        # Acc waist
        temp = dataset[j][3][1][1][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        standing_total_per_row.append(pd.DataFrame(temp))
        # Acc thigh
        temp = dataset[j][3][1][4][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        standing_total_per_row.append(pd.DataFrame(temp))
        # Acc head
        temp = dataset[j][3][1][5][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        standing_total_per_row.append(pd.DataFrame(temp))

        # Gyr forearm
        temp = dataset[j][3][2][1][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        standing_total_per_row.append(pd.DataFrame(temp))
        # Gyr waist
        temp = dataset[j][3][2][6][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        standing_total_per_row.append(pd.DataFrame(temp))
        # Gyr thigh
        temp = dataset[j][3][2][3][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        standing_total_per_row.append(pd.DataFrame(temp))
        # Gyr head
        temp = dataset[j][3][2][2][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        standing_total_per_row.append(pd.DataFrame(temp))

        # Mag forearm
        temp = dataset[j][3][3][1][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        standing_total_per_row.append(pd.DataFrame(temp))
        # Mag waist
        temp = dataset[j][3][3][6][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        standing_total_per_row.append(pd.DataFrame(temp))
        # Mag thigh
        temp = dataset[j][3][3][4][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        standing_total_per_row.append(pd.DataFrame(temp))
        # Mag head
        temp = dataset[j][3][3][2][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        standing_total_per_row.append(pd.DataFrame(temp))

        if i == j:
            standing_test = pd.concat(standing_total_per_row, axis = 1, ignore_index = True)
        else:
            standing_total.append(pd.concat(standing_total_per_row, axis = 1, ignore_index = True))
    standing = pd.concat(standing_total, axis = 0, ignore_index = True)
    standing.dropna(inplace = True)
    standing_test.dropna(inplace = True)
    standing["label"] = "standing"
    standing_test["label"] = "standing"
    # print(standing.shape)

    # Walking Dataframe from dataset
    walking_total = []
    walking_test = []
    for j in range(15):
        if j == 1:
            continue
        walking_total_per_row = []
        # Acc forearm
        temp = dataset[j][4][1][5][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        walking_total_per_row.append(pd.DataFrame(temp))
        # Acc waist
        temp = dataset[j][4][1][3][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        walking_total_per_row.append(pd.DataFrame(temp))
        # Acc thigh
        temp = dataset[j][4][1][0][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        walking_total_per_row.append(pd.DataFrame(temp))
        # Acc head
        temp = dataset[j][4][1][2][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        walking_total_per_row.append(pd.DataFrame(temp))

        # Gyr forearm
        temp = dataset[j][4][2][4][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        walking_total_per_row.append(pd.DataFrame(temp))
        # Gyr waist
        temp = dataset[j][4][2][3][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        walking_total_per_row.append(pd.DataFrame(temp))
        # Gyr thigh
        temp = dataset[j][4][2][6][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        walking_total_per_row.append(pd.DataFrame(temp))
        # Gyr head
        temp = dataset[j][4][2][2][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        walking_total_per_row.append(pd.DataFrame(temp))

        # Mag forearm
        temp = dataset[j][4][3][1][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        walking_total_per_row.append(pd.DataFrame(temp))
        # Mag waist
        temp = dataset[j][4][3][6][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        walking_total_per_row.append(pd.DataFrame(temp))
        # Mag thigh
        temp = dataset[j][4][3][4][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        walking_total_per_row.append(pd.DataFrame(temp))
        # Mag head
        temp = dataset[j][4][3][2][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        walking_total_per_row.append(pd.DataFrame(temp))

        if i == j:
            walking_test = pd.concat(walking_total_per_row, axis = 1, ignore_index = True)
        else:
            walking_total.append(pd.concat(walking_total_per_row, axis = 1, ignore_index = True))
    walking = pd.concat(walking_total, axis = 0, ignore_index = True)
    walking.dropna(inplace = True)
    walking_test.dropna(inplace = True)
    walking["label"] = "walking"
    walking_test["label"] = "walking"
    # print(walking.shape)

    # Climbing down Dataframe from dataset
    climbing_down_total = []
    climbing_down_test = []
    for j in range(15):
        if j == 1:
            continue
        climbing_down_total_per_row = []
        # Acc forearm
        temp = dataset[j][5][1][3][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_down_total_per_row.append(pd.DataFrame(temp))
        # Acc waist
        temp = dataset[j][5][1][5][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_down_total_per_row.append(pd.DataFrame(temp))
        # Acc thigh
        temp = dataset[j][5][1][4][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_down_total_per_row.append(pd.DataFrame(temp))
        # Acc head
        temp = dataset[j][5][1][2][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_down_total_per_row.append(pd.DataFrame(temp))

        # Gyr forearm
        temp = dataset[j][5][2][0][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_down_total_per_row.append(pd.DataFrame(temp))
        # Gyr waist
        temp = dataset[j][5][2][1][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_down_total_per_row.append(pd.DataFrame(temp))
        # Gyr thigh
        temp = dataset[j][5][2][3][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_down_total_per_row.append(pd.DataFrame(temp))
        # Gyr head
        temp = dataset[j][5][2][5][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_down_total_per_row.append(pd.DataFrame(temp))

        # Mag forearm
        temp = dataset[j][5][3][1][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_down_total_per_row.append(pd.DataFrame(temp))
        # Mag waist
        temp = dataset[j][5][3][6][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_down_total_per_row.append(pd.DataFrame(temp))
        # Mag thigh
        temp = dataset[j][5][3][4][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_down_total_per_row.append(pd.DataFrame(temp))
        # Mag head
        temp = dataset[j][5][3][2][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_down_total_per_row.append(pd.DataFrame(temp))


        if i == j:
            climbing_down_test = pd.concat(climbing_down_total_per_row, axis = 1, ignore_index = True)
        else:
            climbing_down_total.append(pd.concat(climbing_down_total_per_row, axis = 1, ignore_index = True))
    climbing_down = pd.concat(climbing_down_total, axis = 0, ignore_index = True)
    climbing_down.dropna(inplace = True)
    climbing_down_test.dropna(inplace = True)
    climbing_down["label"] = "climbing_down"
    climbing_down_test["label"] = "climbing_down"
    # print(climbing_down.shape)

    # Climbing up Dataframe from dataset
    climbing_up_total = []
    climbing_up_test = []
    for j in range(15):
        if j == 1:
            continue
        climbing_up_total_per_row = []
        # Acc forearm
        temp = dataset[j][6][1][6][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_up_total_per_row.append(pd.DataFrame(temp))
        # Acc waist
        temp = dataset[j][6][1][4][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_up_total_per_row.append(pd.DataFrame(temp))
        # Acc thigh
        temp = dataset[j][6][1][0][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_up_total_per_row.append(pd.DataFrame(temp))
        # Acc head
        temp = dataset[j][6][1][1][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_up_total_per_row.append(pd.DataFrame(temp))

        # Gyr forearm
        temp = dataset[j][6][2][2][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_up_total_per_row.append(pd.DataFrame(temp))
        # Gyr waist
        temp = dataset[j][6][2][1][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_up_total_per_row.append(pd.DataFrame(temp))
        # Gyr thigh
        temp = dataset[j][6][2][5][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_up_total_per_row.append(pd.DataFrame(temp))
        # Gyr head
        if j == 3 or j == 6 or j == 13:
            temp = dataset[j][6][2][4][1]      # Gyrooscope_climbingup_thigh
        else:
            temp = dataset[j][6][2][3][1]      # Gyrooscope_climbingup_shin
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_up_total_per_row.append(pd.DataFrame(temp))

        # Mag forearm
        temp = dataset[j][6][3][1][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_up_total_per_row.append(pd.DataFrame(temp))
        # Mag waist
        temp = dataset[j][6][3][6][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_up_total_per_row.append(pd.DataFrame(temp))
        # Mag thigh
        temp = dataset[j][6][3][4][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_up_total_per_row.append(pd.DataFrame(temp))
        # Mag head
        temp = dataset[j][6][3][2][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        climbing_up_total_per_row.append(pd.DataFrame(temp))

        if i == j:
            climbing_up_test = pd.concat(climbing_up_total_per_row, axis = 1, ignore_index = True)
        else:
            climbing_up_total.append(pd.concat(climbing_up_total_per_row, axis = 1, ignore_index = True))
    climbing_up = pd.concat(climbing_up_total, axis = 0, ignore_index = True)
    climbing_up.dropna(inplace = True)
    climbing_up_test.dropna(inplace = True)
    climbing_up["label"] = "climbing_up"
    climbing_up_test["label"] = "climbing_up"
    # print(climbing_up.shape)

    # Jumping Dataframe from dataset
    jumping_total = []
    jumping_test = []
    for j in range(15):
        if j == 1:
            continue
        jumping_total_per_row = []
        # Acc forearm
        temp = dataset[j][7][1][0][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        jumping_total_per_row.append(pd.DataFrame(temp))
        # Acc waist
        temp = dataset[j][7][1][3][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        jumping_total_per_row.append(pd.DataFrame(temp))
        # Acc thigh
        temp = dataset[j][7][1][5][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        jumping_total_per_row.append(pd.DataFrame(temp))
        # Acc head
        temp = dataset[j][7][1][4][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        jumping_total_per_row.append(pd.DataFrame(temp))

        # Gyr forearm
        temp = dataset[j][7][2][2][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        jumping_total_per_row.append(pd.DataFrame(temp))
        # Gyr waist
        temp = dataset[j][7][2][3][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        jumping_total_per_row.append(pd.DataFrame(temp))
        # Gyr thigh
        temp = dataset[j][7][2][0][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        jumping_total_per_row.append(pd.DataFrame(temp))
        # Gyr head
        temp = dataset[j][7][2][1][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        jumping_total_per_row.append(pd.DataFrame(temp))

        # Mag forearm
        temp = dataset[j][7][3][1][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        jumping_total_per_row.append(pd.DataFrame(temp))
        # Mag waist
        # temp = dataset[j][7][3][6][1] the real one. WHY OUT OF INDEX????
        temp = dataset[j][7][3][4][1]   # this is Mag thigh. I just wanted to ignore this error "Out of bound" when I put temp = dataset[j][7][3][6][1]  (Mag Waist)
        #print("HIEEEEER", temp)
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        jumping_total_per_row.append(pd.DataFrame(temp))
        # Mag thigh
        temp = dataset[j][7][3][4][1]
        #print("HIEEEEER2", temp)
        #if i == 0:       #commented because "['id' 'attr_time'] not found in axis". temp = [4325 rows x 3 columns] = (attr_x  attr_y  attr_z)
            #temp.drop(['id', 'attr_time'], axis = 1, inplace = True). Cus I already droped it for Mag waist.
        jumping_total_per_row.append(pd.DataFrame(temp))
        # Mag head
        temp = dataset[j][7][3][2][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        jumping_total_per_row.append(pd.DataFrame(temp))

        if i == j:
            jumping_test = pd.concat(jumping_total_per_row, axis = 1, ignore_index = True)
        else:
            jumping_total.append(pd.concat(jumping_total_per_row, axis = 1, ignore_index = True))
    jumping = pd.concat(jumping_total, axis = 0, ignore_index = True)
    jumping.dropna(inplace = True)
    jumping_test.dropna(inplace = True)
    jumping["label"] = "jumping"
    jumping_test["label"] = "jumping"
    # print(jumping.shape)

    # Lying Dataframe from dataset
    lying_total = []
    lying_test = []
    for j in range(15):
        if j == 1:
            continue
        lying_total_per_row = []
        # Acc forearm
        temp = dataset[j][8][1][4][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        lying_total_per_row.append(pd.DataFrame(temp))
        # Acc waist
        temp = dataset[j][8][1][0][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        lying_total_per_row.append(pd.DataFrame(temp))
        # Acc thigh
        temp = dataset[j][8][1][2][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        lying_total_per_row.append(pd.DataFrame(temp))
        # Acc head
        temp = dataset[j][8][1][5][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        lying_total_per_row.append(pd.DataFrame(temp))

        # Gyr forearm
        temp = dataset[j][8][2][5][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        lying_total_per_row.append(pd.DataFrame(temp))
        # Gyr waist
        temp = dataset[j][8][2][3][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        lying_total_per_row.append(pd.DataFrame(temp))
        # Gyr thigh
        temp = dataset[j][8][2][1][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        lying_total_per_row.append(pd.DataFrame(temp))
        # Gyr head
        temp = dataset[j][8][2][6][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        lying_total_per_row.append(pd.DataFrame(temp))

        # Mag forearm
        temp = dataset[j][8][3][1][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        lying_total_per_row.append(pd.DataFrame(temp))
        # Mag waist
        temp = dataset[j][8][3][6][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        lying_total_per_row.append(pd.DataFrame(temp))
        # Mag thigh
        temp = dataset[j][8][3][4][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        lying_total_per_row.append(pd.DataFrame(temp))
        # Mag head
        temp = dataset[j][8][3][2][1]
        if i == 0:
            temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
        lying_total_per_row.append(pd.DataFrame(temp))


        if i == j:
            lying_test = pd.concat(lying_total_per_row, axis = 1, ignore_index = True)
        else:
            lying_total.append(pd.concat(lying_total_per_row, axis = 1, ignore_index = True))
    lying = pd.concat(lying_total, axis = 0, ignore_index = True)
    lying.dropna(inplace = True)
    lying_test.dropna(inplace = True)
    lying["label"] = "lying"
    lying_test["label"] = "lying"
    # print(jumping.shape)

    all_activities_total_train_subjects = [running, sitting, standing, walking,
                                           climbing_down, climbing_up, jumping, lying]
    size = all_activities_total_train_subjects[0].shape[0]
    for j in range(1, len(all_activities_total_train_subjects)):
        if size > all_activities_total_train_subjects[j].shape[0]:
            size = all_activities_total_train_subjects[j].shape[0]
    for j in range(0, len(all_activities_total_train_subjects)):
        all_activities_total_train_subjects[j] = all_activities_total_train_subjects[j].iloc[:size, :]
        print(all_activities_total_train_subjects[j].shape)
    all_activities_train_subjects = pd.concat(all_activities_total_train_subjects, axis = 0, ignore_index = True)

    activity_windows_train_subjects, activity_indices_train_subjects \
        = sw.sliding_window_samples(all_activities_train_subjects, 50, 25)

    all_activities_total_test_subjects = [running_test, sitting_test, standing_test, walking_test,
                                          climbing_down_test, climbing_up_test, jumping_test, lying_test]
    all_activities_test_subjects = pd.concat(all_activities_total_test_subjects, axis = 0, ignore_index = True)

    start = time.time()
    activity_windows_test_subjects, activity_indices_test_subjects \
        = sw.sliding_window_samples(all_activities_test_subjects, 50, 25)
    print("Sliding window done")
    get_function_duration(start)

    start = time.time()
    x_train_before_shuffle, y_train_before_shuffle = do_windows_slicing(activity_windows_train_subjects)
    x_test, y_test = do_windows_slicing(activity_windows_test_subjects)
    print("Slicing done")
    get_function_duration(start)

    start = time.time()
    x_train_before_shuffle, y_train_before_shuffle = calculate_features(x_train_before_shuffle, y_train_before_shuffle)
    x_test, y_test = calculate_features(x_test, y_test)
    print("Feature Calculation done in : ")
    get_function_duration(start)

    x_train, x_something, y_train, y_something = train_test_split(x_train_before_shuffle, y_train_before_shuffle,
                                                                  test_size = 0.01, random_state = 42)

    print("Starting Classification")

    start = time.time()
    knn = KNeighborsClassifier(n_neighbors = 12)
    knn.fit(x_train, y_train)
    y_pred_knn = knn.predict(x_test)
    all_accuracies = get_accuracy(y_test, y_pred_knn, "KNN")
    generate_confusion_matrix(knn, x_test, y_test, "K Nearest Neighbor for subject ", i, all_accuracies)
    get_function_duration(start)

    start = time.time()
    svc = make_pipeline(StandardScaler(), SVC(gamma = 'auto'))
    svc.fit(x_train, y_train)
    y_pred_svc = svc.predict(x_test)
    all_accuracies = get_accuracy(y_test, y_pred_svc, "SVC")
    generate_confusion_matrix(svc, x_test, y_test, "Support Vector Machine for subject ", i, all_accuracies)
    get_function_duration(start)

    start = time.time()
    d_tree = DecisionTreeClassifier(random_state = 0)
    d_tree.fit(x_train, y_train)
    y_pred_d_tree = d_tree.predict(x_test)
    all_accuracies = get_accuracy(y_test, y_pred_d_tree, "DTree")
    generate_confusion_matrix(d_tree, x_test, y_test, "Decision Tree for subject ", i, all_accuracies)
    get_function_duration(start)

    start = time.time()
    ada_boost = AdaBoostClassifier(n_estimators = 50, random_state = 0)
    ada_boost.fit(x_train, y_train)
    y_pred_ada_boost = ada_boost.predict(x_test)
    all_accuracies = get_accuracy(y_test, y_pred_ada_boost, "AdaBoost")
    generate_confusion_matrix(ada_boost, x_test, y_test, "Adaboost for subject ", i, all_accuracies)
    get_function_duration(start)

    start = time.time()
    ada_d_tree = AdaBoostClassifier(DecisionTreeClassifier(random_state = 0), n_estimators = 50, random_state = 0)
    ada_d_tree.fit(x_train, y_train)
    y_pred_ada_d_tree = ada_d_tree.predict(x_test)
    all_accuracies = get_accuracy(y_test, y_pred_ada_d_tree, "AdaDTree")
    generate_confusion_matrix(ada_d_tree, x_test, y_test, "Adaboost Decision Tree for subject ", i, all_accuracies)
    get_function_duration(start)

    start = time.time()
    gaussian_naive_bayes = GaussianNB()
    gaussian_naive_bayes.fit(x_train, y_train)
    y_pred_gnb = gaussian_naive_bayes.predict(x_test)
    all_accuracies = get_accuracy(y_test, y_pred_gnb, "GNB")
    generate_confusion_matrix(gaussian_naive_bayes, x_test, y_test, "Gaussian Naive Bayes for subject ", i, all_accuracies)
    get_function_duration(start)

    start = time.time()
    rfc = RandomForestClassifier(max_depth = 6, random_state = 0)
    rfc.fit(x_train, y_train)
    y_pred_rfc = rfc.predict(x_test)
    all_accuracies = get_accuracy(y_test, y_pred_rfc, "RFC")
    generate_confusion_matrix(rfc, x_test, y_test, "Random Forest Classifier for subject ", i, all_accuracies)
    get_function_duration(start)

    start = time.time()
    mlp = MLPClassifier(hidden_layer_sizes = (100, 80, 60, 40, 20, 10),
                        max_iter = 1000,
                        activation = "identity")
    mlp.fit(x_train, y_train)
    y_pred_mlp = mlp.predict(x_test)
    all_accuracies = get_accuracy(y_test, y_pred_mlp, "MLP")
    generate_confusion_matrix(mlp, x_test, y_test, "Multilayer Perceptron for subject ", i, all_accuracies)
    get_function_duration(start)

    get_function_duration(single_subject_time)

get_function_duration(all_subjects_time)
