from processing.docs_processor import Processor
from preparation.docs_reader import Reader
from modeling.lda import LDA

documents_path = '/home/adrian/Unibuc/Probabilistic Programming/Topic-Modeling-System/data/raw'
number_of_topics = 5

documents = Reader.read_documents(documents_path)

processor = Processor("english")
sorted_vocabulary, word_id, id_word, vocabulary_size = processor.create_vocabulary(documents)

data = Processor.generate_data_from_documents(documents, vocabulary_size, word_id)
lda_model = LDA(data, number_of_topics, vocabulary_size)
lda_model.fit()

lda_model.plot_word_distribution()
lda_model.show_topic_words(id_word)
