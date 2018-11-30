import numpy as np


class Processor(object):

    @staticmethod
    def generate_data_temp(number_of_documents, number_of_words):
        return np.random.randint(0, 10, size=(number_of_documents, number_of_words))
