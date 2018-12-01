from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import numpy as np
from array import array

class Processor(object):

    def __init__(self, stopwords_language):
        # nltk.download('stopwords')
        # nltk.download('punkt')

        self.word_frequency = 0
        self.stopwords = set(stopwords.words(stopwords_language))

    @staticmethod
    def generate_data_temp(number_of_documents, number_of_words):
        return np.random.randint(0, 10, size=(number_of_documents, number_of_words))

    def create_vocabulary(self, documents):
        vocabulary = dict()

        for document in documents:
            for word in word_tokenize(document):
                word = word.lower()
                if word not in self.stopwords and word.isalpha():
                    if word not in vocabulary:
                        vocabulary[word] = 0
                    vocabulary[word] += 1

        # sorted_vocabulary = filter(lambda x: x[1] > self.word_frequency, sorted(vocabulary.items(), key=lambda x: x[1], reverse=True))
        sorted_vocabulary = filter(lambda x: x[1] > self.word_frequency, Processor.do_nothing(vocabulary.items()))
        # sorted_vocabulary_2 = filter(lambda x: x[1] > self.word_frequency, sorted(vocabulary.items(), key=lambda x: x[1], reverse=True))
        sorted_vocabulary_2 = filter(lambda x: x[1] > self.word_frequency, Processor.do_nothing(vocabulary.items()))

        word_id = {v[0]: i for i, v in enumerate(sorted_vocabulary)}
        id_word = {i: v[0] for i, v in enumerate(sorted_vocabulary_2)}
        vocabulary_size = len(word_id)

        return sorted_vocabulary, word_id, id_word, vocabulary_size

    @staticmethod
    def do_nothing(my_list):
        return my_list

    @staticmethod
    def generate_data_from_documents(documents, word_ids):
        data = []

        for document in documents:
            local_data = []
            for word in word_tokenize(document):
                word = word.lower()
                if word in word_ids:
                    local_data.append(word_ids[word])

            data.append(local_data)

        return data
