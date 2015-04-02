from abc import abstractmethod

__author__ = 'finde'


class IBM_Model(object):
    def __init__(self):
        self.param_names = ()
        self.params = {}

    def get_param(self, param_name):
        """Returns a container of parameters with the given name."""
        return self.params(param_name)

    def train(self, language_pair):
        """Trains the IBM model using the given list of language pair."""
        pass

    def predict(self):
        pass