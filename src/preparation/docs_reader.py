from os import listdir
from os.path import isfile, join


class Reader(object):

    @staticmethod
    def read(path):
        """
        Open and read a file.
        :param path: path of file
        :return: content of file
        """
        return open(path, 'r').read()

    @staticmethod
    def list_of_documents(directory):
        """
        Create a list of file from a specified directory.
        :param directory: the directory path
        :return: a list of files founded in directory
        """
        return [file for file in listdir(directory) if isfile(join(directory, file))]

    @staticmethod
    def read_documents(directory):
        """
        Read content for each text file from a specified directory
        :param directory: the directory path
        :return: a list of documents content
        """
        documents = sorted(Reader.list_of_documents(directory))
        return [Reader.read(directory + "/" + document) for document in documents]
