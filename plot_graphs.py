# Import datasets, classifiers and performance metrics
from sklearn import datasets, svm, metrics
import pdb
import argparse

from utils import (
    preprocess_digits,
    train_dev_test_split,
    h_param_tuning,
    data_viz,
    pred_image_viz,
    get_all_h_param_comb,
    tune_and_save,
)
from joblib import dump, load

classifier_default = "svm"
random_default = 1

parser = argparse.ArgumentParser(description = "no description")
parser.add_argument("--clf_name",default = classifier_default,help = "no help")
parser.add_argument("--random_state",default = random_default,help = "no help")
args = parser.parse_args()

classifier = args.clf_name
random_state = int(args.random_state)

print("this is " , classifier)
print("this is " , random_state)

train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
assert train_frac + dev_frac + test_frac == 1.0

# 1. set the ranges of hyper parameters
gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

params = {}
params["gamma"] = gamma_list
params["C"] = c_list

h_param_comb = get_all_h_param_comb(params)


print("h param is ")
print(h_param_comb)

# PART: load dataset -- data from csv, tsv, jsonl, pickle
digits = datasets.load_digits()
data_viz(digits)
data, label = preprocess_digits(digits)
# housekeeping
del digits


x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
    data, label, train_frac, dev_frac,random_state
)

# PART: Define the model
# Create a classifier: a support vector classifier

if classifier == 'svm':
    clf = svm.SVC()

elif classifier == "tree":
    clf = tree.DecisionTreeClassifier()
# define the evaluation metric
metric = metrics.accuracy_score



actual_model_path = tune_and_save(
    clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path=None
)




# 2. load the best_model
#best_model = load(actual_model_path)

# PART: Get test set predictions
# Predict the value of the digit on the test subset
#predicted = best_model.predict(x_test)

#print(metrics.classification_report(y_test, predicted))

#pred_image_viz(x_test, predicted)

# 4. report the test set accurancy with that best model.
# PART: Compute evaluation metrics
#print(
#    f"Classification report for classifier {clf}:\n"
#    f"{metrics.classification_report(y_test, predicted)}\n"
#)

