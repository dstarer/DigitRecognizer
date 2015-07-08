__author__ = 'admin'

import ConfigParser

class Configuration:
    def __init__(self, configuration_file="configuration.ini"):
        parser = ConfigParser.RawConfigParser()
        parser.read(configuration_file)

        self.parameters = dict({'train':{}, 'predict':{}})

        self.parameters['train'].setdefault('select', parser.getint("train", "select"))
        self.parameters['train'].setdefault('train_csv', parser.get('train', 'train_csv'))
        self.parameters['train'].setdefault('test_csv', parser.get('train', 'test_csv'))
        self.parameters['train'].setdefault('algorithm', parser.get('train', 'algorithm'))
        self.parameters['train'].setdefault('model', parser.get('train', 'model'))
        self.parameters['train'].setdefault('n_estimators', parser.getint('train', 'n_estimators'))
        self.parameters['train'].setdefault('svm_svc', parser.get('train', 'svm_svc'))

        self.parameters['predict'].setdefault('select', parser.getint('predict', 'select'))
        self.parameters['predict'].setdefault('model', parser.get('predict', 'model'))
        self.parameters['predict'].setdefault('test_csv', parser.get('predict', 'test_csv'))
        self.parameters['predict'].setdefault('predict_result', parser.get('predict', 'predict_result'))

    def get_train_parameter(self, attribute):
        assert self.parameters.get('train').has_key(attribute)
        return self.parameters.get('train').get(attribute)

    def get_predict_parameter(self, attribute):
        assert self.parameters.get('predict').has_key(attribute)
        return self.parameters.get('predict').get(attribute)

    def get_train_paramters(self):
        return self.parameters.get('train')

    def get_predict_parameters(self):
        return self.parameters.get('predict')

    def is_train_alive(self):
        return self.parameters.get('train').get('select') == 1

    def is_predict_alive(self):
        return self.parameters.get('predict').get('select') == 1


if __name__ == "__main__":
    conf = Configuration()
    print conf.parameters