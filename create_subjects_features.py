import statistics

import sliding_window as sw
import read_dataset as rd
import pandas as pd
import numpy as np
import time

from sklearn.preprocessing import StandardScaler


def get_function_duration(start_time):
    end = time.time()
    hours, rem = divmod(end - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds), "\n")


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
    for index in range(len(x)):
        features = pd.DataFrame(x[index])
        total_x_out = []
        for features_index in range(features.shape[1]):

            # Uncomment this snippet of code for Cross-Correlation and comment the normal fft in the next line
            # temp_calculate_features_fourier = np.absolute(
            #     np.fft.ifft(
            #         np.fft.fft(features[features_index].to_numpy())
            #         * np.fft.fft(np.flip(features[features_index].to_numpy()))
            #     )
            # )

            # Comment and uncomment only the next line when needed
            temp_calculate_features_fourier = np.absolute(np.fft.fft(features[features_index].to_numpy()))

            temp_calculate_features_mean = statistics.mean(features[features_index].to_numpy())
            temp_calculate_features_std = statistics.stdev(features[features_index].to_numpy())
            x_out = np.insert(temp_calculate_features_fourier[:int(len(temp_calculate_features_fourier) * 0.1)],
                              0,
                              [temp_calculate_features_mean])
            x_out = np.insert(x_out, 0, [temp_calculate_features_std])
            total_x_out.append(x_out)
        total_x_out = np.concatenate(total_x_out)
        x[index] = total_x_out
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    pd_x_y = pd.concat([x, y], axis = 1)
    pd_x_y.dropna(inplace = True)
    new_x = pd_x_y.iloc[:, 0:pd_x_y.shape[1] - 1]
    new_y = pd_x_y.iloc[:, pd_x_y.shape[1] - 1]
    new_x = StandardScaler().fit_transform(new_x)
    # print(new_x)
    return new_x, new_y


def remove_unwanted_columns(temp_df):
    if not {'id', 'attr_time'}.issubset(temp_df.columns):
        return temp_df
    else:
        temp_df.drop(['id', 'attr_time'], axis = 1, inplace = True)
        return temp_df


start = time.time()
print("Started Execution")
path = "Dataset"
dataset = rd.get_all_subjects_data(path)
print("Data extracted")
get_function_duration(start)

matches = ["shin", "waist", "thigh", "head", "chest", "upperarm"]

all_subjects_time = time.time()
for i in range(15):
    # if i == 1 or i == 5 or i == 6:
    #     continue
    print("subject ", i)
    single_subject_time = time.time()

    # Running Dataframe from dataset
    running_total_per_row = []

    """ sensor_index loops through the sensors in the Dataset Dataframe
    body_part_index loops through the body parts where the sensors were placed
    and looks for the matches wanted in the matches list in the beginning of the code
    once the body part is found, it is then appended to the running_total_per_row list """
    for sensor_index in range(len(dataset[i][1])):
        for body_part_index in range(len(dataset[i][1][sensor_index])):
            if any(x in dataset[i][1][sensor_index][body_part_index][0] for x in matches):
                temp = dataset[i][1][sensor_index][body_part_index][1]
                running_total_per_row.append(pd.DataFrame(remove_unwanted_columns(temp)))

    running = pd.concat(running_total_per_row, axis = 1, ignore_index = True)
    running.dropna(inplace = True)
    running["label"] = "running"
    # print(running.shape)

    # Sitting Dataframe from dataset
    sitting_total_per_row = []

    """ sensor_index loops through the sensors in the Dataset Dataframe
    body_part_index loops through the body parts where the sensors were placed
    and looks for the matches wanted in the matches list in the beginning of the code
    once the body part is found, it is then appended to the sitting_total_per_row list """
    for sensor_index in range(len(dataset[i][2])):
        for body_part_index in range(len(dataset[i][2][sensor_index])):
            if any(x in dataset[i][2][sensor_index][body_part_index][0] for x in matches):
                temp = dataset[i][2][sensor_index][body_part_index][1]
                sitting_total_per_row.append(pd.DataFrame(remove_unwanted_columns(temp)))

    sitting = pd.concat(sitting_total_per_row, axis = 1, ignore_index = True)
    sitting.dropna(inplace = True)
    sitting["label"] = "sitting"
    # print(sitting.shape)

    # Standing Dataframe from dataset
    standing_total_per_row = []

    """ sensor_index loops through the sensors in the Dataset Dataframe
    body_part_index loops through the body parts where the sensors were placed
    and looks for the matches wanted in the matches list in the beginning of the code
    once the body part is found, it is then appended to the standing_total_per_row list """
    for sensor_index in range(len(dataset[i][3])):
        for body_part_index in range(len(dataset[i][3][sensor_index])):
            if any(x in dataset[i][3][sensor_index][body_part_index][0] for x in matches):
                temp = dataset[i][3][sensor_index][body_part_index][1]
                standing_total_per_row.append(pd.DataFrame(remove_unwanted_columns(temp)))

    standing = pd.concat(standing_total_per_row, axis = 1, ignore_index = True)
    standing.dropna(inplace = True)
    standing["label"] = "standing"
    # print(standing.shape)

# Walking Dataframe from dataset
    walking_total_per_row = []

    """ sensor_index loops through the sensors in the Dataset Dataframe
    body_part_index loops through the body parts where the sensors were placed
    and looks for the matches wanted in the matches list in the beginning of the code
    once the body part is found, it is then appended to the walking_total_per_row list """
    for sensor_index in range(len(dataset[i][4])):
        for body_part_index in range(len(dataset[i][4][sensor_index])):
            if any(x in dataset[i][4][sensor_index][body_part_index][0] for x in matches):
                temp = dataset[i][4][sensor_index][body_part_index][1]
                walking_total_per_row.append(pd.DataFrame(remove_unwanted_columns(temp)))

    walking = pd.concat(walking_total_per_row, axis = 1, ignore_index = True)
    walking.dropna(inplace = True)
    walking["label"] = "walking"
    # print(walking.shape)

# Climbing down Dataframe from dataset
    climbing_down_total_per_row = []

    """ sensor_index loops through the sensors in the Dataset Dataframe
    body_part_index loops through the body parts where the sensors were placed
    and looks for the matches wanted in the matches list in the beginning of the code
    once the body part is found, it is then appended to the climbing_down_total_per_row list """
    for sensor_index in range(len(dataset[i][5])):
        for body_part_index in range(len(dataset[i][5][sensor_index])):
            if any(x in dataset[i][5][sensor_index][body_part_index][0] for x in matches):
                temp = dataset[i][5][sensor_index][body_part_index][1]
                climbing_down_total_per_row.append(pd.DataFrame(remove_unwanted_columns(temp)))

    climbing_down = pd.concat(climbing_down_total_per_row, axis = 1, ignore_index = True)
    climbing_down.dropna(inplace = True)
    climbing_down["label"] = "climbing_down"
    # print(climbing_down.shape)

    # Climbing up Dataframe from dataset
    climbing_up_total_per_row = []

    """ sensor_index loops through the sensors in the Dataset Dataframe
    body_part_index loops through the body parts where the sensors were placed
    and looks for the matches wanted in the matches list in the beginning of the code
    once the body part is found, it is then appended to the climbing_up_total_per_row list """
    for sensor_index in range(len(dataset[i][6])):
        for body_part_index in range(len(dataset[i][6][sensor_index])):
            if any(x in dataset[i][6][sensor_index][body_part_index][0] for x in matches):
                temp = dataset[i][6][sensor_index][body_part_index][1]
                climbing_up_total_per_row.append(pd.DataFrame(remove_unwanted_columns(temp)))

    climbing_up = pd.concat(climbing_up_total_per_row, axis = 1, ignore_index = True)
    climbing_up.dropna(inplace = True)
    climbing_up["label"] = "climbing_up"
    # print(climbing_up.shape)

    # Jumping Dataframe from dataset
    jumping_total_per_row = []

    """ sensor_index loops through the sensors in the Dataset Dataframe
    body_part_index loops through the body parts where the sensors were placed
    and looks for the matches wanted in the matches list in the beginning of the code
    once the body part is found, it is then appended to the jumping_total_per_row list """
    for sensor_index in range(len(dataset[i][7])):
        for body_part_index in range(len(dataset[i][7][sensor_index])):
            if any(x in dataset[i][7][sensor_index][body_part_index][0] for x in matches):
                temp = dataset[i][7][sensor_index][body_part_index][1]
                jumping_total_per_row.append(pd.DataFrame(remove_unwanted_columns(temp)))

    jumping = pd.concat(jumping_total_per_row, axis = 1, ignore_index = True)
    jumping.dropna(inplace = True)
    jumping["label"] = "jumping"
    # print(jumping.shape)

    # Lying Dataframe from dataset
    lying_total_per_row = []

    """ sensor_index loops through the sensors in the Dataset Dataframe
    body_part_index loops through the body parts where the sensors were placed
    and looks for the matches wanted in the matches list in the beginning of the code
    once the body part is found, it is then appended to the lying_total_per_row list """
    for sensor_index in range(len(dataset[i][8])):
        for body_part_index in range(len(dataset[i][8][sensor_index])):
            if any(x in dataset[i][8][sensor_index][body_part_index][0] for x in matches):
                temp = dataset[i][8][sensor_index][body_part_index][1]
                lying_total_per_row.append(pd.DataFrame(remove_unwanted_columns(temp)))

    lying = pd.concat(lying_total_per_row, axis = 1, ignore_index = True)
    lying.dropna(inplace = True)
    lying["label"] = "lying"
    # print(jumping.shape)

    print(running.shape)
    print(sitting.shape)
    print(standing.shape)
    print(walking.shape)
    print(climbing_down.shape)
    print(climbing_up.shape)
    print(jumping.shape)
    print(lying.shape)
    print("\n")

    # [running, sitting, standing, walking,
    #  climbing_down, climbing_up, jumping, lying]
    all_activities_single_subject = [running, sitting, standing, walking]

    size = all_activities_single_subject[0].shape[0]
    for j in range(1, len(all_activities_single_subject)):
        if size > all_activities_single_subject[j].shape[0]:
            size = all_activities_single_subject[j].shape[0]

    for j in range(0, len(all_activities_single_subject)):
        remove_n = all_activities_single_subject[j].shape[0] - size
        drop_indices = np.random.choice(all_activities_single_subject[j].index, remove_n, replace = False)
        all_activities_single_subject[j] = all_activities_single_subject[j].drop(drop_indices)
        print(all_activities_single_subject[j].shape)
    all_activities_train_subjects = pd.concat(all_activities_single_subject, axis = 0, ignore_index = True)

    activity_windows_train_subjects, activity_indices_train_subjects \
        = sw.sliding_window_samples(all_activities_train_subjects, 50, 25)

    start = time.time()
    x_train_before_shuffle, y_train_before_shuffle = do_windows_slicing(activity_windows_train_subjects)
    print("Slicing done")
    get_function_duration(start)

    start = time.time()
    x_train_before_shuffle, y_train_before_shuffle = calculate_features(x_train_before_shuffle, y_train_before_shuffle)
    print("Feature Calculation done in : ")
    get_function_duration(start)

    x_train = pd.DataFrame(x_train_before_shuffle)
    y_train = pd.DataFrame(y_train_before_shuffle)

    x_train.to_csv("SubjectsFeatures/subject_" + str(i) + "_x.csv", index = False)
    y_train.to_csv("SubjectsFeatures/subject_" + str(i) + "_y.csv", index = False)

print("Full Execution time: ")
get_function_duration(all_subjects_time)
