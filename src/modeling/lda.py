import numpy as np
import pymc as pm
import matplotlib.pyplot as plt
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

        # TODO: - trebuie verificat -> self.model = pm.Model([self.phi, self.theta, self.Z, self.W])
        self.model = pm.Model([self.theta, self.phi, self.Z, self.W])
        self.mcmc = pm.MCMC(self.model)

    def fit(self, iterations=1000, burn_in=10):
        self.mcmc.sample(iterations, burn_in)

    @staticmethod
    def kl_divergence(p, q):
        """
        TODO:
        :param p:
        :param q:
        :return:
        """
        return np.sum(p * np.log10(p/q))

    @staticmethod
    def cosine_sim(x, y):
        """
        TODO:
        :param x:
        :param y:
        :return:
        """
        return np.dot(x, y) / np.sqrt(np.dot(x, x) * np.dot(y, y))

    def sorted_docs_sim(self):
        kl_divergence_docs = \
            [(i,
              j,
              self.kl_divergence(self.theta[i].value, self.theta[j].value),
              self.cosine_sim(self.data[i], self.data[j])) for i in range(len(self.theta)) for j in range(len(self.theta)) if i != j]

        return sorted(kl_divergence_docs, key=lambda x: x[3], reverse=True)

    def show_topics(self):
        return self.phi.value

    def show_words(self):
        return self.W.value

    def show_topic_words(self, id_words):
        for i, t in enumerate(self.phi.value):
            print("Topic %i : " % i,
                  ", ".join(id_words[w_] for w_ in np.argsort(t[0])[-10:] if w_ < (self.vocabulary - 1 - 1)))

    def plot_data(self):
        plt.clf()
        plt.matshow(self.data, fignum=1000, cmap=plt.cm.Reds)
        plt.gca().set_aspect('auto')
        plt.xlabel("Words")
        plt.ylabel("Documents")

    def plot_words_per_topic(self, ax=None):
        if ax is None:
            plt.clf()
            fig, ax = plt.subplots(1, 1)

        words = self.Z.value
        topic_dist = dict()
        for k_i in words:
            for k in k_i:
                if k not in topic_dist:
                    topic_dist[k] = 0
                topic_dist[k] += 1

        ax.bar(topic_dist.keys(), topic_dist.values())
        ax.set_xlabel("Topics")
        ax.set_ylabel("Counts")
        ax.set_title("Document words per topics")
        plt.show()

    def plot_word_distribution(self, ax=None):
        global figure
        topics = self.phi.value

        if ax is None:
            plt.clf()
            figure, ax = plt.subplots(math.floor((len(topics) + 1) / 2), 2, figsize=(10, 10))

        for i, t in enumerate(topics):
            ax[i / 2][i % 2].bar(range(len(t[0])), t[0])
            ax[i / 2][i % 2].set_title("Topic %s" % i)

        plt.suptitle("Vocab word proportions per topic")
        figure.subplots_adjust(hspace=0.5, wspace=0.5)
