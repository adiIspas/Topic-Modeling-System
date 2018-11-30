from processing.docs_processor import Processor
from modeling.lda import LDA

number_of_documents = 20
number_of_words = 10
number_of_topics = 5

data_temp = Processor.generate_data_temp(number_of_documents, number_of_words)
lda_model = LDA(data_temp, number_of_topics, number_of_words)

lda_model.fit()
