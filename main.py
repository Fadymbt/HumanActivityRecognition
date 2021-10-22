import pandas as pd
import matplotlib.pyplot as plt
import time
import os

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, plot_confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier


def get_accuracy(y, y_pred, algorithm_name):
    f1_macro = "f1 score macro " + algorithm_name + " = " + \
               str(round(f1_score(y, y_pred, average = 'macro') * 100, 4)) + "%"
    f1_micro = "f1 score micro " + algorithm_name + " = " + \
               str(round(f1_score(y, y_pred, average = 'micro') * 100, 4)) + "%"
    acc = "accuracy " + algorithm_name + " = " + \
          str(round(accuracy_score(y, y_pred) * 100, 4)) + "%"
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
    # plot_confusion_matrix(algorithm, x, y)
    ConfusionMatrixDisplay.from_estimator(algorithm, x, y)
    plt.title(algorithm_name + str(index + 1))
    plt.xticks(rotation = 90)
    plt.xlabel("Predicted label\n" + acc)
    plt.tight_layout()
    if not os.path.exists("Output/Subject" + "_" + str(index + 1)):
        os.makedirs("Output/Subject" + "_" + str(index + 1))
    plt.savefig("Output/Subject" + "_" + str(index + 1) + "/" + algorithm_name + str(index + 1) + ".jpg")
    plt.show()
    return plt


def get_subject(subject_number):
    x = pd.read_csv("SubjectsFeatures/subject_" + str(subject_number) + "_x.csv")
    y = pd.read_csv("SubjectsFeatures/subject_" + str(subject_number) + "_y.csv")
    return [x, y]


all_subjects_time = time.time()

all_subjects = []

# Append all subjects to all_subjects array
for i in range(15):
    # if i != 1 and i != 5:
    all_subjects.append(get_subject(i))

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
            x_test = pd.DataFrame(all_subjects[j][0])
            y_test = pd.DataFrame(all_subjects[j][1])
            print("I am testing = ", j)

    print("\n")
    x_train_before_shuffle = pd.concat(x_train_before_shuffle)
    y_train_before_shuffle = pd.concat(y_train_before_shuffle)

    x_train, x_something, y_train, y_something = train_test_split(x_train_before_shuffle, y_train_before_shuffle,
                                                                  test_size = 0.01, random_state = 42)

    y_train = y_train.values.ravel()
    y_test = y_test.values.ravel()

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

    # start = time.time()
    # ada_boost = AdaBoostClassifier(n_estimators = 50, random_state = 0)
    # ada_boost.fit(x_train, y_train)
    # y_pred_ada_boost = ada_boost.predict(x_test)
    # all_accuracies = get_accuracy(y_test, y_pred_ada_boost, "AdaBoost")
    # generate_confusion_matrix(ada_boost, x_test, y_test, "Adaboost for subject ", i, all_accuracies)
    # get_function_duration(start)

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

print("Full Execution time: ")
get_function_duration(all_subjects_time)
