from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class Processor(object):

    def __init__(self, stopwords_language):
        """
        Processor is used to create vocabulary and generate data from documents content.
        :param stopwords_language: the language of documents content
        """

        # Uncomment just you have to download 'stopwords' and 'punkt' from nltk module.
        # nltk.download('stopwords')
        # nltk.download('punkt')

        self.word_frequency = 0  # For large documents increase this variable
        self.stopwords = set(stopwords.words(stopwords_language))

    def create_vocabulary(self, documents):
        """
        Create a vocabulary with unique words from a list of documents content.
        :param documents: the documents content
        :return:
        - :sorted_vocabulary: a list of words with frequency greater than :word_frequency and alphabetically sorted
        - :word_id: is a mapping between words and its ids, example: cat - 0, dog - 1
        - :id_word: is a mapping between ids and its words, example: 0 - cat, 1 - dog
        - :vocabulary_size: the size of vocabulary
        """
        vocabulary = dict()

        for document in documents:
            for word in word_tokenize(document):
                word = word.lower()
                if word not in self.stopwords and word.isalpha():
                    if word not in vocabulary:
                        vocabulary[word] = 0
                    vocabulary[word] += 1

        sorted_vocabulary = filter(lambda x: x[1] > self.word_frequency, sorted(vocabulary.items(), key=lambda x: x[1], reverse=True))
        sorted_vocabulary_2 = filter(lambda x: x[1] > self.word_frequency, sorted(vocabulary.items(), key=lambda x: x[1], reverse=True))

        word_id = {v[0]: i for i, v in enumerate(sorted_vocabulary)}
        id_word = {i: v[0] for i, v in enumerate(sorted_vocabulary_2)}
        vocabulary_size = len(word_id)

        return sorted_vocabulary, word_id, id_word, vocabulary_size

    @staticmethod
    def generate_data_from_documents(documents, word_ids):
        """
        Generate data through mapping each word from each document with its unique id.
        :param documents: a list of documents content
        :param word_ids: the mapping between word and its id
        :return: a list of words ids for each document
            example: [I, like, pizza, and, I, like, donuts] -> [0, 1, 2, 3, 0, 1, 4]
            * stopwords is keeped in final result just for example
        """
        data = []

        for document in documents:
            local_data = []
            for word in word_tokenize(document):
                word = word.lower()
                if word in word_ids:
                    local_data.append(word_ids[word])

            data.append(local_data)

        return data
