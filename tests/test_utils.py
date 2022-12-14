import sys, os
import numpy as np
from joblib import load
from sklearn import datasets, svm, metrics
from utils import (
    preprocess_digits,
    train_dev_test_split,
    h_param_tuning,
    data_viz,
    pred_image_viz,
    get_all_h_param_comb,
    tune_and_save,
)

sys.path.append(".")

from utils import get_all_h_param_comb, tune_and_save
from sklearn import svm, metrics

# test case to check if all the combinations of the hyper parameters are indeed getting created
def test_get_h_param_comb():
    gamma_list = [0.01, 0.005, 0.001, 0.0005, 0.0001]
    c_list = [0.1, 0.2, 0.5, 0.7, 1, 2, 5, 7, 10]

    params = {}
    params["gamma"] = gamma_list
    params["C"] = c_list
    h_param_comb = get_all_h_param_comb(params)

    assert len(h_param_comb) == len(gamma_list) * len(c_list)

def helper_h_params():
    # small number of h params
    gamma_list = [0.01, 0.005]
    c_list = [0.1, 0.2]

    params = {}
    params["gamma"] = gamma_list
    params["C"] = c_list
    h_param_comb = get_all_h_param_comb(params)
    return h_param_comb

def helper_create_bin_data(n=100, d=7):
    x_train_0 = np.random.randn(n, d)
    x_train_1 = 1.5 + np.random.randn(n, d)
    x_train = np.vstack((x_train_0, x_train_1))
    y_train = np.zeros(2 * n)
    y_train[n:] = 1

    return x_train, y_train

def test_tune_and_save():    
    h_param_comb = helper_h_params()
    x_train, y_train = helper_create_bin_data(n=100, d=7)
    x_dev, y_dev = x_train, y_train

    clf = svm.SVC()
    metric = metrics.accuracy_score
    
    model_path = "test_run_model_path.joblib"
    actual_model_path = tune_and_save(clf, x_train, y_train, x_dev, y_dev, metric, h_param_comb, model_path)

    assert actual_model_path == model_path
    assert os.path.exists(actual_model_path)
    assert type(load(actual_model_path)) == type(clf)


def test_bias():

    train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1
    # hyperparameters 

    gamma = 0.001
    c  = 0.5
    params = {"gamma" : gamma , "C" : c}

    # dataset
    digits = datasets.load_digits()
    data, label = preprocess_digits(digits)
    # housekeeping
    del digits
    

    print("number of labels " , len(label))
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac
    )
 
    # model

    clf = svm.SVC()
    clf.set_params(**params)
    clf.fit(x_train,y_train)

    predicted_dev = clf.predict(x_dev)

    #print(predicted_dev)

    num_labels  = 0

    for i in range(len(predicted_dev)-1):
        if predicted_dev[i] != predicted_dev[i+1]:
            num_labels += 1

    print(num_labels)
    
    assert num_labels != 0 


def test_all():

    # hyperparameters 
    train_frac, dev_frac, test_frac = 0.8, 0.1, 0.1

    gamma = 0.001
    c  = 0.5
    params = {"gamma" : gamma , "C" : c}

    # dataset
    digits = datasets.load_digits()
    data, label = preprocess_digits(digits)
    # housekeeping
    del digits
    

    print("number of labels " , len(label))
    x_train, y_train, x_dev, y_dev, x_test, y_test = train_dev_test_split(
        data, label, train_frac, dev_frac
    )
 
    # model

    clf = svm.SVC()
    clf.set_params(**params)
    clf.fit(x_train,y_train)

    predicted_dev = clf.predict(x_dev)

    #print(predicted_dev)

    num_labels  = 0

    for i in range(len(predicted_dev)-1):
        if predicted_dev[i] != predicted_dev[i+1]:
            num_labels += 1

    print(num_labels)

    assert num_labels == len(label) 


# what more test cases should be there
# irrespective of the changes to the refactored code.

# train/dev/test split functionality : input 200 samples, fraction is 70:15:15, then op should have 140:30:30 samples in each set


# preprocessing gives ouput that is consumable by model

# accuracy check. if acc(model) < threshold, then must not be pushed.

# hardware requirement test cases are difficult to write.
# what is possible: (model size in execution) < max_memory_you_support

# latency: tik; model(input); tok == time passed < threshold
# this is dependent on the execution environment (as close the actual prod/runtime environment)


# model variance? --
# bias vs variance in ML ?
# variance is if model changes bit performance might change drastically so it's del accuracy / del train_set

# std([model(train_1), model(train_2), ..., model(train_k)]) < threshold
# i.e if data changes bit , it's performance should not change a lot and therefor variation
# in model performances represented by standard deviation should be inside a limit.


# Data set we can verify, if it as desired
# dimensionality of the data --

# Verify output size, say if you want output in certain way
# assert len(prediction_y) == len(test_y)

# model persistance?
# train the model -- check performance -- write the model to disk
# is the model loaded from the disk same as what we had written?
# assert acc(loaded_model) == expected_acc // acc we have saved earlier for the same model found earlier
# assert predictions (loaded_model) == expected_prediction