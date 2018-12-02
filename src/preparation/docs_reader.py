from os import listdir
from os.path import isfile, join


class Reader(object):

    @staticmethod
    def read(filename):
        return open(filename, 'r').read()

    @staticmethod
    def extract_documents(directory):
        return [file for file in listdir(directory) if isfile(join(directory, file))]

    @staticmethod
    def read_documents(directory):
        documents = sorted(Reader.extract_documents(directory))
        return [Reader.read(directory + "/" + document) for document in documents]
