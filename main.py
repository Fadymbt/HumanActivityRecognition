import statistics
import sliding_window as sw
import read_dataset as rd
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler, Normalizer

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

# running = running.sample(frac = 1).reset_index(drop = True)


def get_accuracy(y, y_pred, algorithm_name):
    print("f1 score macro ", algorithm_name, " = ", f1_score(y, y_pred, average = 'macro'))
    print("f1 score micro ", algorithm_name, " = ", f1_score(y, y_pred, average = 'micro'))
    print("accuracy ", algorithm_name, " = ", accuracy_score(y, y_pred))


def get_function_duration(start_time):
    end = time.time()
    hours, rem = divmod(end - start_time, 3600)
    minutes, seconds = divmod(rem, 60)
    print("{:0>2}:{:0>2}:{:05.2f}".format(int(hours), int(minutes), seconds), "\n")


def generate_confusion_matrix(algorithm, x, y, algorithm_name):
    plot_confusion_matrix(algorithm, x, y)
    plt.title(algorithm_name)
    plt.show()


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
        for j in range(features.shape[1]):
            temp_calculate_features_fourier = np.absolute(np.fft.fft(features[j].to_numpy()))
            temp_calculate_features_mean = statistics.mean(features[j].to_numpy())
            temp_calculate_features_std = statistics.stdev(features[j].to_numpy())
            x_out = np.insert(temp_calculate_features_fourier[:int(len(temp_calculate_features_fourier) * 0.1)], 0, [temp_calculate_features_mean])
            x_out = np.insert(x_out, 0, [temp_calculate_features_std])
            total_x_out.append(x_out)
        total_x_out = np.concatenate(total_x_out)
        x[i] = total_x_out
    x = pd.DataFrame(x)
    y = pd.DataFrame(y)
    pd_x_y = pd.concat([x, y], axis = 1)
    pd_x_y.dropna(inplace = True)
    print(pd_x_y)
    print("x size = ", x.shape)
    print("y size = ", y.shape)
    print("pd_x_y size = ", pd_x_y.shape)
    new_x = pd_x_y.iloc[:, 0:pd_x_y.shape[1] - 1]
    new_y = pd_x_y.iloc[:, pd_x_y.shape[1] - 1]
    print("new x size = ", new_x.shape)
    print("new y size = ", new_y.shape)
    # StandardScaler().fit(new_x)
    # Normalizer().fit(new_x)
    print(new_x)
    return new_x, new_y


start = time.time()
path = "Dataset"
dataset = rd.get_all_subjects_data(path)
print("Data extracted")
get_function_duration(start)


# Running Dataframe from dataset
running_total = []
for i in range(14):
    running_total_per_row = []
    # Acc forearm
    temp = dataset[i][1][1][0][1]
    temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
    running_total_per_row.append(pd.DataFrame(temp))
    # Acc waist
    temp = dataset[i][1][1][6][1]
    temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
    running_total_per_row.append(pd.DataFrame(temp))
    # Acc thigh
    temp = dataset[i][1][1][3][1]
    temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
    running_total_per_row.append(pd.DataFrame(temp))
    # Acc head
    temp = dataset[i][1][1][2][1]
    temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
    running_total_per_row.append(pd.DataFrame(temp))
    running_total.append(pd.concat(running_total_per_row, axis = 1, ignore_index = True))
running = pd.concat(running_total, axis = 0, ignore_index = True)
running.dropna(inplace = True)
running["label"] = "running"
# print(running.shape)

# Get Running Test Data
running_test_total = []
# Acc forearm
temp = dataset[14][1][1][0][1]
temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
running_test_total.append(pd.DataFrame(temp))
# Acc waist
temp = dataset[14][1][1][6][1]
temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
running_test_total.append(pd.DataFrame(temp))
# Acc thigh
temp = dataset[14][1][1][3][1]
temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
running_test_total.append(pd.DataFrame(temp))
# Acc thigh
temp = dataset[14][1][1][2][1]
temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
running_test_total.append(pd.DataFrame(temp))
running_test = pd.concat(running_test_total, axis = 1, ignore_index = True)
running_test.dropna(inplace = True)
running_test["label"] = "running"

# Sitting Dataframe from dataset
sitting_total = []
for i in range(14):
    sitting_total_per_row = []
    # Acc forearm
    temp = dataset[i][2][1][5][1]
    temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
    sitting_total_per_row.append(pd.DataFrame(temp))
    # Acc waist
    temp = dataset[i][2][1][0][1]
    temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
    sitting_total_per_row.append(pd.DataFrame(temp))
    # Acc thigh
    temp = dataset[i][2][1][3][1]
    temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
    sitting_total_per_row.append(pd.DataFrame(temp))
    # Acc head
    temp = dataset[i][2][1][2][1]
    temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
    sitting_total_per_row.append(pd.DataFrame(temp))
    sitting_total.append(pd.concat(sitting_total_per_row, axis = 1, ignore_index = True))
sitting = pd.concat(sitting_total, axis = 0, ignore_index = True)
sitting.dropna(inplace = True)
sitting["label"] = "sitting"
# print(sitting.shape)

# Get Sitting Test Data
sitting_test_total = []
# Acc forearm
temp = dataset[14][2][1][5][1]
temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
sitting_test_total.append(pd.DataFrame(temp))
# Acc waist
temp = dataset[14][2][1][0][1]
temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
sitting_test_total.append(pd.DataFrame(temp))
# Acc thigh
temp = dataset[14][2][1][3][1]
temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
sitting_test_total.append(pd.DataFrame(temp))
# Acc head
temp = dataset[14][2][1][2][1]
temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
sitting_test_total.append(pd.DataFrame(temp))
sitting_test = pd.concat(sitting_test_total, axis = 1, ignore_index = True)
sitting_test.dropna(inplace = True)
sitting_test["label"] = "sitting"

# Standing Dataframe from dataset
standing_total = []
for i in range(14):
    standing_total_per_row = []
    # Acc forearm
    temp = dataset[i][3][1][0][1]
    temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
    standing_total_per_row.append(pd.DataFrame(temp))
    # Acc waist
    temp = dataset[i][3][1][1][1]
    temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
    standing_total_per_row.append(pd.DataFrame(temp))
    # Acc thigh
    temp = dataset[i][3][1][4][1]
    temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
    standing_total_per_row.append(pd.DataFrame(temp))
    # Acc head
    temp = dataset[i][3][1][5][1]
    temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
    standing_total_per_row.append(pd.DataFrame(temp))
    standing_total.append(pd.concat(standing_total_per_row, axis = 1, ignore_index = True))
standing = pd.concat(standing_total, axis = 0, ignore_index = True)
standing.dropna(inplace = True)
standing["label"] = "standing"
# print(standing.shape)

# In the same order as the for loop
standing_test_total = []
# Acc forearm
temp = dataset[14][3][1][0][1]
temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
standing_test_total.append(pd.DataFrame(temp))
# Acc waist
temp = dataset[14][3][1][1][1]
temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
standing_test_total.append(pd.DataFrame(temp))
# Acc thigh
temp = dataset[14][3][1][4][1]
temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
standing_test_total.append(pd.DataFrame(temp))
# Acc head
temp = dataset[14][3][1][5][1]
temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
standing_test_total.append(pd.DataFrame(temp))
standing_test = pd.concat(standing_test_total, axis = 1, ignore_index = True)
standing_test.dropna(inplace = True)
standing_test["label"] = "standing"

# Walking Dataframe from dataset
walking_total = []
for i in range(14):
    walking_total_per_row = []
    # Acc forearm
    temp = dataset[i][4][1][5][1]
    temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
    walking_total_per_row.append(pd.DataFrame(temp))
    # Acc waist
    temp = dataset[i][4][1][3][1]
    temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
    walking_total_per_row.append(pd.DataFrame(temp))
    # Acc thigh
    temp = dataset[i][4][1][0][1]
    temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
    walking_total_per_row.append(pd.DataFrame(temp))
    # Acc head
    temp = dataset[i][4][1][2][1]
    temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
    walking_total_per_row.append(pd.DataFrame(temp))
    walking_total.append(pd.concat(walking_total_per_row, axis = 1, ignore_index = True))
walking = pd.concat(walking_total, axis = 0, ignore_index = True)
walking.dropna(inplace = True)
walking["label"] = "walking"
# print(walking.shape)

# In the same order as the for loop
walking_test_total = []
# Acc forearm
temp = dataset[14][4][1][5][1]
temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
walking_test_total.append(pd.DataFrame(temp))
# Acc waist
temp = dataset[14][4][1][3][1]
temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
walking_test_total.append(pd.DataFrame(temp))
# Acc thigh
temp = dataset[14][4][1][0][1]
temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
walking_test_total.append(pd.DataFrame(temp))
# Acc head
temp = dataset[14][4][1][2][1]
temp.drop(['id', 'attr_time'], axis = 1, inplace = True)
walking_test_total.append(pd.DataFrame(temp))
walking_test = pd.concat(walking_test_total, axis = 1, ignore_index = True)
walking_test.dropna(inplace = True)
walking_test["label"] = "walking"


all_activities_total_train_subjects = [running, sitting, standing, walking]
size = all_activities_total_train_subjects[0].shape[0]
for i in range(1, len(all_activities_total_train_subjects)):
    if size > all_activities_total_train_subjects[i].shape[0]:
        size = all_activities_total_train_subjects[i].shape[0]
for i in range(0, len(all_activities_total_train_subjects)):
    all_activities_total_train_subjects[i] = all_activities_total_train_subjects[i].iloc[:size, :]
    print(all_activities_total_train_subjects[i].shape)
all_activities_train_subjects = pd.concat(all_activities_total_train_subjects, axis = 0, ignore_index = True)

activity_windows_train_subjects, activity_indices_train_subjects \
    = sw.sliding_window_samples(all_activities_train_subjects, 50, 25)


all_activities_total_test_subjects = [running_test, sitting_test, standing_test, walking_test]
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
knn = KNeighborsClassifier(n_neighbors = 16)
knn.fit(x_train, y_train)
y_pred_knn = knn.predict(x_test)
get_accuracy(y_test, y_pred_knn, "KNN")
generate_confusion_matrix(knn, x_test, y_test, "K Nearest Neighbor")
get_function_duration(start)


start = time.time()
svc = make_pipeline(StandardScaler(), SVC(gamma = 'auto'))
svc.fit(x_train, y_train)
y_pred_svc = svc.predict(x_test)
get_accuracy(y_test, y_pred_svc, "SVC")
generate_confusion_matrix(svc, x_test, y_test, "Support Vector Machine")
get_function_duration(start)


start = time.time()
d_tree = DecisionTreeClassifier(random_state = 0)
d_tree.fit(x_train, y_train)
y_pred_d_tree = d_tree.predict(x_test)
get_accuracy(y_test, y_pred_d_tree, "DTree")
generate_confusion_matrix(d_tree, x_test, y_test, "Decision Tree")
get_function_duration(start)


start = time.time()
ada_boost = AdaBoostClassifier(n_estimators = 50, random_state = 0)
ada_boost.fit(x_train, y_train)
y_pred_ada_boost = ada_boost.predict(x_test)
get_accuracy(y_test, y_pred_ada_boost, "AdaBoost")
generate_confusion_matrix(ada_boost, x_test, y_test, "Adaboost")
get_function_duration(start)


start = time.time()
ada_d_tree = AdaBoostClassifier(DecisionTreeClassifier(random_state = 0), n_estimators = 50, random_state = 0)
ada_d_tree.fit(x_train, y_train)
y_pred_ada_d_tree = ada_d_tree.predict(x_test)
get_accuracy(y_test, y_pred_ada_d_tree, "AdaDTree")
generate_confusion_matrix(ada_d_tree, x_test, y_test, "Adaboost Decision Tree")
get_function_duration(start)


start = time.time()
gaussian_naive_bayes = GaussianNB()
gaussian_naive_bayes.fit(x_train, y_train)
y_pred_gnb = gaussian_naive_bayes.predict(x_test)
get_accuracy(y_test, y_pred_gnb, "GNB")
generate_confusion_matrix(gaussian_naive_bayes, x_test, y_test, "Gaussian Naive Bayes")
get_function_duration(start)


start = time.time()
rfc = RandomForestClassifier(max_depth = 6, random_state = 0)
rfc.fit(x_train, y_train)
y_pred_rfc = rfc.predict(x_test)
get_accuracy(y_test, y_pred_rfc, "RFC")
generate_confusion_matrix(rfc, x_test, y_test, "Random Forest Classifier")
get_function_duration(start)


start = time.time()
mlp = MLPClassifier(verbose = True,
                    hidden_layer_sizes = (100, 80, 60, 40, 20, 10),
                    max_iter = 1000,
                    activation = "identity")
mlp.fit(x_train, y_train)
y_pred_mlp = mlp.predict(x_test)
get_accuracy(y_test, y_pred_mlp, "MLP")
generate_confusion_matrix(mlp, x_test, y_test, "Multilayer Perceptron")
get_function_duration(start)
