import os
import pandas as pd


def get_subject_files(subject_path):
    running_files = get_running_files(subject_path + "/running")
    sitting_files = get_sitting_files(subject_path + "/sitting")
    standing_files = get_standing_files(subject_path + "/standing")
    walking_files = get_walking_files(subject_path + "/walking")
    return [subject_path.partition("/")[2], running_files, sitting_files, standing_files, walking_files]


def get_running_files(running_path):
    running_acc_file = get_acc_files(running_path + "/acc_running_csv")
    running_gyr_file = get_gyr_files(running_path + "/gyr_running_csv")
    running_mag_file = get_mag_files(running_path + "/mag_running_csv")
    return ["running", running_acc_file, running_gyr_file, running_mag_file]


def get_sitting_files(sitting_path):
    sitting_acc_file = get_acc_files(sitting_path + "/acc_sitting_csv")
    sitting_gyr_file = get_gyr_files(sitting_path + "/gyr_sitting_csv")
    sitting_mag_file = get_mag_files(sitting_path + "/mag_sitting_csv")
    return ["sitting", sitting_acc_file, sitting_gyr_file, sitting_mag_file]


def get_standing_files(standing_path):
    standing_acc_file = get_acc_files(standing_path + "/acc_standing_csv")
    standing_gyr_file = get_gyr_files(standing_path + "/gyr_standing_csv")
    standing_mag_file = get_mag_files(standing_path + "/mag_standing_csv")
    return ["standing", standing_acc_file, standing_gyr_file, standing_mag_file]


def get_walking_files(walking_path):
    walking_acc_file = get_acc_files(walking_path + "/acc_walking_csv")
    walking_gyr_file = get_gyr_files(walking_path + "/gyr_walking_csv")
    walking_mag_file = get_mag_files(walking_path + "/mag_walking_csv")
    return ["walking", walking_acc_file, walking_gyr_file, walking_mag_file]


def get_acc_files(acc_path):
    return read_csv_files(acc_path)


def get_gyr_files(gyr_path):
    return read_csv_files(gyr_path)


def get_mag_files(mag_path):
    return read_csv_files(mag_path)


def read_csv_files(final_path):
    all_files = []
    for root, dirs, files in os.walk(final_path):
        for file in files:
            if file != "readMe":
                csv_file = pd.read_csv(final_path + "/" + file)
                all_files.append([file[:-4], csv_file])
    return all_files


def get_all_subjects_data(path):
    subject_1_data = get_subject_files(path + "/subject_1")
    subject_2_data = get_subject_files(path + "/subject_2")
    subject_3_data = get_subject_files(path + "/subject_3")
    subject_4_data = get_subject_files(path + "/subject_4")
    return [subject_1_data, subject_2_data, subject_3_data, subject_4_data]


# {
#   "subject_1": [
#     "running": [
#       "acc_running_forearm":[id, attr_time, attr_x, attr_y, attr_z],
#       "acc_running_chest":[id, attr_time, attr_x, attr_y, attr_z]
#       ],
#     "sitting": [
#       "acc_sitting_forearm":[id, attr_time, attr_x, attr_y, attr_z]
#       ],
#     "standing": [
#       ],
#     "Walking":[
#       ]
#   ]
#   "subject_2": []
#   "subject_3": []
# }
