import pandas as pd
import matplotlib.pyplot as plt
import time
import os

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


activity_dictionary = {
    "running": 1.0,
    "sitting": 2.0,
    "standing": 3.0,
    "walking": 4.0,
    "climbing_down": 5.0,
    "climbing_up": 6.0,
    "jumping": 7.0,
    "lying": 8.0
}


def get_accuracy(y, y_pred, algorithm_name):
    f1_macro = "f1 score macro " + algorithm_name + " = " + \
               str(round(f1_score(y, y_pred, average = 'macro') * 100, 2)) + "%"
    f1_micro = "f1 score micro " + algorithm_name + " = " + \
               str(round(f1_score(y, y_pred, average = 'micro') * 100, 2)) + "%"
    acc = "accuracy " + algorithm_name + " = " + \
          str(round(accuracy_score(y, y_pred) * 100, 2)) + "%"
    print(f1_macro)
    print(f1_micro)
    print(acc)
    return f1_macro + "\n" + f1_micro + "\n" + acc


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


def convert_strings_to_floats(y):
    for y_key in range(len(y)):
        y[y_key] = activity_dictionary[y[y_key]]
    return y


all_subjects_time = time.time()

subject_0_x = pd.read_csv("SubjectsFeatures/subject_0_x.csv")
subject_0_y = pd.read_csv("SubjectsFeatures/subject_0_y.csv")
subject_0 = [subject_0_x, subject_0_y]

subject_2_x = pd.read_csv("SubjectsFeatures/subject_2_x.csv")
subject_2_y = pd.read_csv("SubjectsFeatures/subject_2_y.csv")
subject_2 = [subject_2_x, subject_2_y]

subject_3_x = pd.read_csv("SubjectsFeatures/subject_3_x.csv")
subject_3_y = pd.read_csv("SubjectsFeatures/subject_3_y.csv")
subject_3 = [subject_3_x, subject_3_y]

subject_4_x = pd.read_csv("SubjectsFeatures/subject_4_x.csv")
subject_4_y = pd.read_csv("SubjectsFeatures/subject_4_y.csv")
subject_4 = [subject_4_x, subject_4_y]

subject_6_x = pd.read_csv("SubjectsFeatures/subject_6_x.csv")
subject_6_y = pd.read_csv("SubjectsFeatures/subject_6_y.csv")
subject_6 = [subject_6_x, subject_6_y]

subject_7_x = pd.read_csv("SubjectsFeatures/subject_7_x.csv")
subject_7_y = pd.read_csv("SubjectsFeatures/subject_7_y.csv")
subject_7 = [subject_7_x, subject_7_y]

subject_8_x = pd.read_csv("SubjectsFeatures/subject_8_x.csv")
subject_8_y = pd.read_csv("SubjectsFeatures/subject_8_y.csv")
subject_8 = [subject_8_x, subject_8_y]

subject_9_x = pd.read_csv("SubjectsFeatures/subject_9_x.csv")
subject_9_y = pd.read_csv("SubjectsFeatures/subject_9_y.csv")
subject_9 = [subject_9_x, subject_9_y]

subject_10_x = pd.read_csv("SubjectsFeatures/subject_10_x.csv")
subject_10_y = pd.read_csv("SubjectsFeatures/subject_10_y.csv")
subject_10 = [subject_10_x, subject_10_y]

subject_11_x = pd.read_csv("SubjectsFeatures/subject_11_x.csv")
subject_11_y = pd.read_csv("SubjectsFeatures/subject_11_y.csv")
subject_11 = [subject_11_x, subject_11_y]

subject_12_x = pd.read_csv("SubjectsFeatures/subject_12_x.csv")
subject_12_y = pd.read_csv("SubjectsFeatures/subject_12_y.csv")
subject_12 = [subject_12_x, subject_12_y]

subject_13_x = pd.read_csv("SubjectsFeatures/subject_13_x.csv")
subject_13_y = pd.read_csv("SubjectsFeatures/subject_13_y.csv")
subject_13 = [subject_13_x, subject_13_y]

subject_14_x = pd.read_csv("SubjectsFeatures/subject_14_x.csv")
subject_14_y = pd.read_csv("SubjectsFeatures/subject_14_y.csv")
subject_14 = [subject_14_x, subject_14_y]

all_subjects = [
    subject_0,
    subject_2,
    subject_3,
    subject_4,
    subject_6,
    subject_7,
    subject_8,
    subject_9,
    subject_10,
    subject_11,
    subject_12,
    subject_13,
    subject_14
]

for i in range(len(all_subjects)):
    single_subject_time = time.time()
    x_train_before_shuffle = []
    y_train_before_shuffle = []
    x_test = []
    y_test = []
    for j in range(len(all_subjects)):
        if i != j:
            x_train_before_shuffle.append(pd.DataFrame(all_subjects[j][0]))
            y_train_before_shuffle.append(pd.DataFrame(all_subjects[j][1]))
            print("I am training = ", j)
        else:
            print("I am testing = ", i)
            x_test = pd.DataFrame(all_subjects[j][0])
            y_test = pd.DataFrame(all_subjects[j][1])

    x_train_before_shuffle = pd.concat(x_train_before_shuffle)
    y_train_before_shuffle = pd.concat(y_train_before_shuffle)

    x_train, x_something, y_train, y_something = train_test_split(x_train_before_shuffle, y_train_before_shuffle,
                                                                  test_size = 0.01, random_state = 42)

    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()
    # print(x_train.shape)
    # print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)
    # y_train.reset_index(drop = True, inplace = True)
    # y_test.reset_index(drop = True, inplace = True)

    print("Starting Classification")

    start = time.time()
    knn = KNeighborsClassifier(n_neighbors = 16)
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
    generate_confusion_matrix(gaussian_naive_bayes, x_test, y_test,
                              "Gaussian Naive Bayes for subject ", i, all_accuracies)
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
