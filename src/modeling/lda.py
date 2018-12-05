import numpy as np
import pymc as pm
import math


class LDA(object):

    def __init__(self, data, topics, vocabulary):
        """
        LDA Model
        :param data: Documents to analyze
        :param topics: Number of topics
        :param vocabulary: Number of words from vocabulary
        """
        self.data = data
        self.topics = topics
        self.vocabulary = vocabulary

        self.docs = len(self.data)
        self.alpha = np.ones(self.topics)
        self.beta = np.ones(self.vocabulary)

        # Word distribution for each topics
        self.phi = pm.Container(
            [pm.CompletedDirichlet("phi_%s" % i,
                                   pm.Dirichlet("pphi_%s" % i, theta=self.beta)) for i in range(self.topics)])

        # Topic distribution for each document
        self.theta = pm.Container(
            [pm.CompletedDirichlet("theta_%s" % i,
                                   pm.Dirichlet("ptheta_%s" % i, theta=self.alpha)) for i in range(self.docs)])

        # Number of words from each document
        self.wd = [len(doc) for doc in self.data]

        # Select a topic for each word from each document
        self.Z = pm.Container(
            [pm.Categorical("z_%s" % d,
                            p=self.theta[d],
                            size=self.wd[d],
                            value=np.random.randint(self.topics, size=self.wd[d])) for d in range(self.docs)])

        # Word distribution associated with selected topic
        self.W = pm.Container(
            [pm.Categorical("w_%s,%s" % (d, i),
                            p=pm.Lambda("phi_z_%s_%s" % (d, i), lambda z=self.Z[d][i], phi=self.phi: phi[z]),
                            value=self.data[d][i], observed=True) for d in range(self.docs) for i in range(self.wd[d])])

        # Create the model
        self.model = pm.Model([self.theta, self.phi, self.Z, self.W])
        self.mcmc = pm.MCMC(self.model)

    def fit(self, iterations=1000, burn_in=10):
        """
        Fit the defined model.
        :param iterations: number of iterations to do, default 1000
        :param burn_in: variables will not be tallied until this many iterations are complete, default 10
        :return: a fitted model
        """
        self.mcmc.sample(iterations, burn_in)

    def show_words(self):
        """
        Show words from fitted model.
        :return: a list of words defined by its ids
        """
        return self.W.value

    def show_topic_words(self, id_words):
        """
        For each topic show a list of representative words.
        :param id_words: the mapping from ids to its words
        :return: topic number and its representative words
        """
        for i, t in enumerate(self.phi.value):
            print("Topic %i: " % i,
                  ", ".join(id_words[w_] for w_ in np.argsort(t[0])[-10:] if w_ < (self.vocabulary - 1 - 1)))

    def show_document_topics(self):
        """
        For each document show a list of representative topics.
        :return: documents with a list of representative topics
        """
        return self.theta.value

    def show_word_distribution_in_topics(self):
        """
        For each topic assign a probability for each word to represent this topic.
        :return: topics with a list of words probabilities
        """
        return self.phi.value

    def show_topic_for_word_in_document(self):
        """
        Show most probably topic of word W in document D.
        :return: a list of topics for each word in each document
        """
        return self.Z.value

    def get_topics_words(self, id_word):
        """
        For each topics get a list of representative words.
        This method should be used in API call.
        :param id_word: the mapping from ids to its words
        :return: topics with its representative words
        """
        topics = dict()
        for i, t in enumerate(self.phi.value):
            topics.update({i: [id_word[w_] for w_ in np.argsort(t[0])[-10:] if w_ < (self.vocabulary - 1 - 1)]})

        return topics

    def get_documents_topics(self, threshold):
        """
        For each documents get a list of representative topics that has a weight > :threshold.
        This method should be used in API call.
        :param threshold: minimum weight for each topic per document
        :return: a list of documents with associated topics
        """
        documents = dict()
        for d, t in enumerate(self.theta.value):
            for v in t:
                documents.update({d: list(idx for idx in range(0, len(v)) if v[idx] > threshold)})

        return documents

    def documents_similarity(self, threshold):
        """
        Compute similarity between document
        :param threshold: the minimum value for two documents to be considered similarly
        :return: similarity score between two documents for each pair of two documents with a similarity score > :threshold
        """
        similarities = []
        for d1, t1 in enumerate(self.theta.value):
            for d2, t2 in enumerate(self.theta.value):
                if d1 != d2:
                    similarities.append([d1, d2, 1 - LDA.__hellinger_distance(t1, t2)])

        similarities = [similarity for similarity in similarities if similarity[2] > threshold]
        return similarities


    @staticmethod
    def __hellinger_distance(topics_document_1, topics_document_2):
        """
        Compute Hellinger distance.
        :param topics_document_1: a list of topics from document 1
        :param topics_document_2: a list of topics from document 2
        :return: distance between documents based on its topics
        """
        score = 0
        for idx in range(len(topics_document_1[0])):
            score = score + pow((math.sqrt(topics_document_1[0][idx]) - (math.sqrt(topics_document_2[0][idx]))), 2)

        return score
