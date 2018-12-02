from processing.docs_processor import Processor
from preparation.docs_reader import Reader
from modeling.lda import LDA

# Init variables and read documents
documents_path = '/home/adrian/Unibuc/Probabilistic Programming/Topic-Modeling-System/data/raw'
documents = Reader.read_documents(documents_path)
number_of_topics = 5

# Create a Processor instance with documents content language and create vocabulary from documents
processor = Processor("english")
sorted_vocabulary, word_id, id_word, vocabulary_size = processor.create_vocabulary(documents)

# Processing documents content in a useful format
data = Processor.generate_data_from_documents(documents, word_id)

# Create a LDA model instance and fit the model
lda_model = LDA(data, number_of_topics, vocabulary_size)
lda_model.fit()

# Show results
print("\n\n === BASE PROJECT ===")
print("\n\n --- Data --- \n")
print(*data, sep="\n")

print("\n\n --- Show topic words --- \n")
lda_model.show_topic_words(id_word)

print("\n\n --- Documents topics --- \n")
print(*lda_model.show_document_topics(), sep="\n")

print("\n\n --- Topic for word W in document D --- \n")
print(*lda_model.show_topic_for_word_in_document(), sep="\n")

print("\n\n --- Distribution of words in topics --- \n")
print(*lda_model.show_word_distribution_in_topics(), sep="\n")


print("\n\n === EXTRAS ===")
print("\n\n --- Similarity of documents --- \n")
print(*lda_model.documents_similarity(0.8), sep="\n")
