import statistics
import sliding_window as sw
import read_dataset as rd
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


path = "Dataset"
dataset = rd.get_all_subjects_data(path)


def do_windows_slicing(activity_windows):
    y = []
    x = []
    for windows in activity_windows:
        window = pd.DataFrame(windows)
        y.append(window.iloc[:, 5].value_counts().idxmax())
        window.drop([0, 5], axis = 1, inplace = True)
        window.reset_index(drop = True, inplace = True)
        numpy_window = window.to_numpy()
        x.append(numpy_window)
        return x, y


def calculate_features(x):
    for i in range(len(x)):
        temp_x = pd.DataFrame(x[i])
        x_axis_before_fourier = temp_x[1]
        y_axis_before_fourier = temp_x[2]
        z_axis_before_fourier = temp_x[3]

        x_axis = np.absolute(np.fft.fft(x_axis_before_fourier))
        y_axis = np.absolute(np.fft.fft(y_axis_before_fourier))
        z_axis = np.absolute(np.fft.fft(z_axis_before_fourier))

        x_mean = np.array(statistics.mean(x_axis_before_fourier))
        y_mean = np.array(statistics.mean(y_axis_before_fourier))
        z_mean = np.array(statistics.mean(z_axis_before_fourier))

        x_std = np.array(statistics.stdev(x_axis_before_fourier))
        y_std = np.array(statistics.stdev(y_axis_before_fourier))
        z_std = np.array(statistics.stdev(z_axis_before_fourier))

        # x_out = [x_mean, y_mean, z_mean, x_std, y_std, z_std]
        x_out = np.concatenate((x_axis[:len(x_axis * 0.1)], y_axis[:len(x_axis * 0.1)], z_axis[:len(x_axis * 0.1)]))
        np.append(x_out, [x_mean, y_mean, z_mean, x_std, y_std, z_std])
        x[i] = x_out
        return x

# Running Dataframe from dataset
sub1_running_acc_forearm_df = dataset[0][1][1][0][1]
sub2_running_acc_forearm_df = dataset[1][1][1][0][1]
sub3_running_acc_forearm_df = dataset[2][1][1][0][1]

sub4_running_acc_forearm_df = dataset[3][1][1][0][1]

# Sitting Dataframe from dataset
sub1_sitting_acc_forearm_df = dataset[0][2][1][5][1]
sub2_sitting_acc_forearm_df = dataset[1][2][1][5][1]
sub3_sitting_acc_forearm_df = dataset[2][2][1][5][1]

sub4_sitting_acc_forearm_df = dataset[3][2][1][5][1]

# Standing Dataframe from dataset
sub1_standing_acc_forearm_df = dataset[0][3][1][0][1]
sub2_standing_acc_forearm_df = dataset[1][3][1][0][1]
sub3_standing_acc_forearm_df = dataset[2][3][1][0][1]

sub4_standing_acc_forearm_df = dataset[3][3][1][0][1]

# Walking Dataframe from dataset
sub1_walking_acc_forearm_df = dataset[0][4][1][5][1]
sub2_walking_acc_forearm_df = dataset[1][4][1][5][1]
sub3_walking_acc_forearm_df = dataset[2][4][1][5][1]

sub4_walking_acc_forearm_df = dataset[3][4][1][5][1]

running_1 = pd.DataFrame(sub1_running_acc_forearm_df)
running_2 = pd.DataFrame(sub2_running_acc_forearm_df)
running_3 = pd.DataFrame(sub3_running_acc_forearm_df)
running_total = [running_1, running_2, running_3]
running = pd.concat(running_total, axis = 0, ignore_index = True)
running["label"] = "running"
# print(running)

sitting_1 = pd.DataFrame(sub1_sitting_acc_forearm_df)
sitting_2 = pd.DataFrame(sub2_sitting_acc_forearm_df)
sitting_3 = pd.DataFrame(sub3_sitting_acc_forearm_df)
sitting_total = [sitting_1, sitting_2, sitting_3]
sitting = pd.concat(sitting_total, axis = 0, ignore_index = True)
sitting["label"] = "sitting"
# print(sitting)

standing_1 = pd.DataFrame(sub1_standing_acc_forearm_df)
standing_2 = pd.DataFrame(sub2_standing_acc_forearm_df)
standing_3 = pd.DataFrame(sub3_standing_acc_forearm_df)
standing_total = [standing_1, standing_2, standing_3]
standing = pd.concat(standing_total, axis = 0, ignore_index = True)
standing["label"] = "standing"
# print(standing)

walking_1 = pd.DataFrame(sub1_walking_acc_forearm_df)
walking_2 = pd.DataFrame(sub2_walking_acc_forearm_df)
walking_3 = pd.DataFrame(sub3_walking_acc_forearm_df)
walking_total = [walking_1, walking_2, walking_3]
walking = pd.concat(walking_total, axis = 0, ignore_index = True)
walking["label"] = "walking"
# print(walking)

all_activities_total_train_subjects = [running, sitting, standing, walking]
all_activities_train_subjects = pd.concat(all_activities_total_train_subjects, axis = 0, ignore_index = True)

activity_windows_train_subjects, activity_indices_train_subjects = sw.sliding_window_samples(all_activities_train_subjects, 50, 25)


all_activities_total_test_subjects = [sub4_running_acc_forearm_df, sub4_sitting_acc_forearm_df, sub4_standing_acc_forearm_df, sub4_walking_acc_forearm_df]
all_activities_test_subjects = pd.concat(all_activities_total_train_subjects, axis = 0, ignore_index = True)

activity_windows_test_subjects, activity_indices_test_subjects = sw.sliding_window_samples(all_activities_train_subjects, 50, 25)


x_train, y_train = do_windows_slicing(pd.DataFrame(activity_windows_train_subjects))

x_test, y_test = do_windows_slicing(pd.DataFrame(activity_windows_test_subjects))

print(x_test)
print(x_train)

x_train = calculate_features(x_train)
x_test = calculate_features(x_test)

#
# x_train, x_test, y_train, y_test \
#     = train_test_split(x, y, test_size = 0.25, random_state = 100)

knn = KNeighborsClassifier(n_neighbors = 4)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
print(f1_score(y_test, y_pred_knn, average='macro'))
print(f1_score(y_test, y_pred_knn, average='micro'))
print("acc KNN = ", accuracy_score(y_test, y_pred_knn), "\n")

svc = make_pipeline(StandardScaler(), SVC(gamma = 'auto'))
svc.fit(x_train, y_train)
y_pred_svc = svc.predict(x_test)
print(f1_score(y_test, y_pred_svc, average='macro'))
print(f1_score(y_test, y_pred_svc, average='micro'))
print("acc SVC = ", accuracy_score(y_test, y_pred_svc), "\n")

d_tree = DecisionTreeClassifier(random_state = 0)
d_tree.fit(x_train, y_train)
y_pred_d_tree = d_tree.predict(x_test)
print(f1_score(y_test, y_pred_d_tree, average='macro'))
print(f1_score(y_test, y_pred_d_tree, average='micro'))
print("acc DTree = ", accuracy_score(y_test, y_pred_d_tree), "\n")

ada_boost = AdaBoostClassifier(n_estimators = 50, random_state = 0)
ada_boost.fit(x_train, y_train)
y_pred_ada_boost = ada_boost.predict(x_test)
print(f1_score(y_test, y_pred_ada_boost, average='macro'))
print(f1_score(y_test, y_pred_ada_boost, average='micro'))
print("acc AdaBoost = ", accuracy_score(y_test, y_pred_ada_boost), "\n")

ada_d_tree = AdaBoostClassifier(DecisionTreeClassifier(random_state = 0), n_estimators = 50, random_state = 0)
ada_d_tree.fit(x_train, y_train)
y_pred_ada_d_tree = ada_d_tree.predict(x_test)
print(f1_score(y_test, y_pred_ada_d_tree, average='macro'))
print(f1_score(y_test, y_pred_ada_d_tree, average='micro'))
print("acc AdaDTree = ", accuracy_score(y_test, y_pred_ada_d_tree), "\n")

gaussian_naive_bayes = GaussianNB()
gaussian_naive_bayes.fit(x_train, y_train)
y_pred_gnb = gaussian_naive_bayes.predict(x_test)
print(f1_score(y_test, y_pred_gnb, average='macro'))
print(f1_score(y_test, y_pred_gnb, average='micro'))
print("acc GNB = ", accuracy_score(y_test, y_pred_gnb), "\n")

rfc = RandomForestClassifier(max_depth = 6, random_state = 0)
rfc.fit(x_train, y_train)
y_pred_rfc = rfc.predict(x_test)
print(f1_score(y_test, y_pred_rfc, average='macro'))
print(f1_score(y_test, y_pred_rfc, average='micro'))
print("acc RFC = ", accuracy_score(y_test, y_pred_rfc), "\n")

mlp = MLPClassifier(hidden_layer_sizes = 1000, max_iter = 100)
mlp.fit(x_train, y_train)
y_pred_mlp = mlp.predict(x_test)
print(f1_score(y_test, y_pred_mlp, average='macro'))
print(f1_score(y_test, y_pred_mlp, average='micro'))
print("acc MLP = ", accuracy_score(y_test, y_pred_mlp), "\n")
