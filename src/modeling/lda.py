import numpy as np
import pymc as pm


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

    def get_topics_words(self, id_words):
        """
        For each topics get a list of representative words.
        This method should be used in API call.
        :param id_words: the mapping from ids to its words
        :return: topics with its representative words
        """
        topics = []
        for i, t in enumerate(self.phi.value):
            topics.append([id_words[w_] for w_ in np.argsort(t[0])[-10:] if w_ < (self.vocabulary - 1 - 1)])

        return topics

    def get_documents_topics(self, min_percent):
        """
        For each documents get a list of representative topics that has a weight > :min_percent.
        This method should be used in API call.
        :param min_percent: minimum weight for each topic per document
        :return: a list of documents with associated topics
        """
        documents = []
        for d, t in enumerate(self.theta.value):
            for v in t:
                documents.append(list(idx for idx in range(0, len(v)) if v[idx] > min_percent))

        return documents
