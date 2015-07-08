__author__ = 'admin'

import numpy as np
from sklearn import svm
from sklearn import metrics
from sklearn.cross_validation import train_test_split
import digit_io
from sklearn.externals import joblib
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from Configuration import Configuration

def check(data, specify_number):
    ## check data dimension
    if data.ndim != 1 and data.shape[1] != 1:
        raise "data dim error!"
    ## make sure that there is at least one sample of specify_number
    if np.sum(data==specify_number) == 0:
        return False

    return True

def split(X, y):

    data_train, data_test, target_train, target_test = \
    train_test_split(X, y)
    successful = False
    while not successful:
        successful = True
        for number in range(10):
            if not check(target_train, number):
                successful = False
                data_train, data_test, target_train, target_test = \
                    train_test_split(X, y)
                break

    return data_train, data_test, target_train, target_test

def get_specific_number(data_set, number):

    assert data_set.ndim == 2

    indices = data_set[0::, 0] == number

    return data_set[indices]

def data_split(data_set):
    train_list = []

    test_list = []

    for number in range(10):

        specific_number = get_specific_number(data_set, number)

        assert specific_number.ndim == 2

        n_train, n_test = train_test_split(specific_number, random_state=42)

        train_list.extend(n_train.tolist())

        test_list.extend(n_test.tolist())

        print '%d passed' % number

    train_set = np.array(train_list)

    test_set = np.array(test_list)

    return train_set, test_set

def test(classifier, data_test, target_test):
    predicted = classifier.predict(data_test)

    print("Classification report for classifier %s: \n%s\n" %\
          (classifier, metrics.classification_report(target_test, predicted)))

    print("Confusion matrix: \n%s" % (metrics.confusion_matrix(target_test, predicted)))

def persist_model(model, filename):
    joblib.dump(model, filename)
    # pickle.dumps(model, open(filename, 'wb'))

def load_model(filename):
    model = joblib.load(filename)
    # model = pickle.load(open(filename, 'rb'))
    return model


def split_entity_dataset():

    train_csv_path = "../data/train.csv"

    data_set = digit_io.read_from_csv(train_csv_path)

    train_set, test_set = data_split(data_set)

    digit_io.write_to_csv(train_set, '../data/training/train.csv')

    digit_io.write_to_csv(test_set, '../data/training/test.csv')

def extract_small_dataset():

    train_csv_path = "../data/train.csv"

    data_set = digit_io.read_from_csv(train_csv_path)

    train_set, test_set = data_split(data_set)

    small_train, small_test = data_split(test_set)

    digit_io.write_to_csv(small_train, "../data/training/small_train.csv")

    digit_io.write_to_csv(small_test, "../data/training/small_test.csv")

def learn_model(classifier, data_train, data_test, target_train, target_test, filename = None):

    # classifier = train(data_train, target_train)
    # classifier = svm.SVC(kernel='linear')

    classifier.fit(data_train, target_train)

    print 'pass train'

    if filename != None:
        persist_model(classifier, filename)

    print 'persist model finished'

    test(classifier, data_test, target_test)

    print 'pass test'

    print 'finished'

def predict(model_file, test_file, filename) :

    classifier = load_model(model_file)

    test = digit_io.read_from_csv(test_file)

    predicted = classifier.predict(test)

    if filename != None:
        # digit_io.write_to_csv(predicted, filename, header=None)
        digit_io.write_predicted_result(predicted, filename)
    else:
        print predicted

def train_model(classifier, train_csv, test_csv, model_file):

    train_set = digit_io.read_from_csv(train_csv)

    test_set = digit_io.read_from_csv(test_csv)

    assert train_set.ndim == 2

    assert test_set.ndim == 2

    data_train, target_train = train_set[0::, 1::], train_set[0::, 0]

    test_train, target_test = test_set[0::, 1::], test_set[0::, 0]

    # classifier = RandomForestClassifier(n_estimators=100)

    learn_model(classifier, data_train, test_train, target_train, target_test, model_file)

def process():

    conf = Configuration()

    if conf.is_train_alive():
        classifier = None
        algorithm = conf.get_train_parameter("algorithm")
        if algorithm == 'randomforest':
            classifier = RandomForestClassifier(conf.get_train_parameter('n_estimators'))
        elif algorithm == 'svm':
            classifier = svm.SVC(kernel=conf.get_train_parameter('svm_svc'))
        elif algorithm == 'bayes':
            classifier = GaussianNB()
        elif algorithm == 'decisiontree':
            classifier = DecisionTreeClassifier()
        else:
            raise "unsupported algorithm!"

        train_model(classifier, conf.get_train_parameter('train_csv'), conf.get_train_parameter('test_csv'), conf.get_train_parameter('model'))

    if conf.is_predict_alive():
        predict(conf.get_predict_parameter('model'), conf.get_predict_parameter('test_csv'), conf.get_predict_parameter('predict_result'))

if "__main__" == __name__:

    # split_entity_dataset()

    # extract_small_dataset()

    process()

