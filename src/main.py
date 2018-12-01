from processing.docs_processor import Processor
from preparation.docs_reader import Reader
from modeling.lda import LDA

documents_path = '/home/adrian/Unibuc/Probabilistic Programming/Topic-Modeling-System/data/raw'
number_of_topics = 2

documents = Reader.read_documents(documents_path)

processor = Processor("english")
sorted_vocabulary, word_id, id_word, vocabulary_size = processor.create_vocabulary(documents)

data = Processor.generate_data_from_documents(documents, word_id)
lda_model = LDA(data, number_of_topics, vocabulary_size)
lda_model.fit()

print("\n\n\n --- Show topic words --- \n")
lda_model.show_topic_words(id_word)

print("\n\n\n --- Documents topics --- \n")
print(*lda_model.show_document_topics(), sep="\n")

print("\n\n\n --- Topic for word W in document D --- \n")
print(*lda_model.show_topic_for_word_in_document(), sep="\n")

print("\n\n\n --- Distribution of words in topics --- \n")
print(*lda_model.show_word_distribution_in_topics(), sep="\n")
