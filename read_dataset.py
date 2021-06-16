import os
import pandas as pd


def get_subject_files(subject_path):
    running_files = get_running_files(subject_path + "/running")
    sitting_files = get_sitting_files(subject_path + "/sitting")
    standing_files = get_standing_files(subject_path + "/standing")
    walking_files = get_walking_files(subject_path + "/walking")
    climbing_down_files = get_climbing_down_files(subject_path + "/climbing_down")
    climbing_up_files = get_climbing_up_files(subject_path + "/climbing_up")
    jumping_files = get_jumping_files(subject_path + "/jumping")
    lying_files = get_lying_files(subject_path + "/lying")
    return [subject_path.partition("/")[2], running_files, sitting_files, standing_files,
            walking_files, climbing_down_files, climbing_up_files, jumping_files, lying_files]


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


def get_climbing_down_files(climbing_down_path):
    climbing_down_acc_file = get_acc_files(climbing_down_path + "/acc_climbingdown_csv")
    climbing_down_gyr_file = get_gyr_files(climbing_down_path + "/gyr_climbingdown_csv")
    climbing_down_mag_file = get_mag_files(climbing_down_path + "/mag_climbingdown_csv")
    return ["climbing_down", climbing_down_acc_file, climbing_down_gyr_file, climbing_down_mag_file]


def get_climbing_up_files(climbing_up_path):
    climbing_up_acc_file = get_acc_files(climbing_up_path + "/acc_climbingup_csv")
    climbing_up_gyr_file = get_gyr_files(climbing_up_path + "/gyr_climbingup_csv")
    climbing_up_mag_file = get_mag_files(climbing_up_path + "/mag_climbingup_csv")
    return ["climbing_up", climbing_up_acc_file, climbing_up_gyr_file, climbing_up_mag_file]


def get_jumping_files(jumping_path):
    jumping_acc_file = get_acc_files(jumping_path + "/acc_jumping_csv")
    jumping_gyr_file = get_gyr_files(jumping_path + "/gyr_jumping_csv")
    jumping_mag_file = get_mag_files(jumping_path + "/mag_jumping_csv")
    return ["jumping", jumping_acc_file, jumping_gyr_file, jumping_mag_file]


def get_lying_files(lying_path):
    lying_acc_file = get_acc_files(lying_path + "/acc_lying_csv")
    lying_gyr_file = get_gyr_files(lying_path + "/gyr_lying_csv")
    lying_mag_file = get_mag_files(lying_path + "/mag_lying_csv")
    return ["lying", lying_acc_file, lying_gyr_file, lying_mag_file]


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
    all_subjects = []
    for i in range(1, 16):
        subject = get_subject_files(path + "/subject_" + str(i))
        all_subjects.append(subject)
    return all_subjects


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
