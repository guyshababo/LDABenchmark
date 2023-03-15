from typing import Set
import re
import os
import xlsxwriter
import gensim
import codecs
import gensim.corpora
from gensim.models import CoherenceModel
import numpy as np
import time


STOP_CHARS = re.compile('[.!?:;]…!')
#STOP_CHARS = re.compile('。')
CHARS_TO_REMOVE = re.compile(r'[\d\n○,.:;\/\[\]\(\)]+')
COLLECTION_PATH = ".\collection"
DO_COHERENCE = False
EXCEL_FILE = '.\\theta_'
#STOP_WORD_FILE = ".\\stopwords\\english_stopwords_2permille.txt"
STOP_WORD_FILE = r"C:\Users\guysh\PycharmProjects\LDABenchmark\stopwords\english_stopwords_all.txt"
CORPUS_FILE = r"C:\Users\guysh\PycharmProjects\LDABenchmark\english_raw.txt"

def jaccard_similarity(topic_1, topic_2):
    """
    Derives the Jaccard similarity of two topics

    Jaccard similarity:
    - A statistic used for comparing the similarity and diversity of sample sets
    - J(A,B) = (A ∩ B)/(A ∪ B)
    - Goal is low Jaccard scores for coverage of the diverse elements
    """
    intersection = set(topic_1).intersection(set(topic_2))
    union = set(topic_1).union(set(topic_2))

    return float(len(intersection)) / float(len(union))

def flatten(l):
    return [item for sublist in l for item in sublist]

def dump_text_as_file(textTitle: str, text: str):
    filename = os.path.join(COLLECTION_PATH, textTitle)
    with codecs.open(filename,
                     mode='w',
                     encoding="utf-8") as MyFile:
        MyFile.write(text)


def read_stopwords(filename: str):
    with codecs.open(filename,
                     mode='rb',
                     encoding="utf-8") as MyFile:
#                     ) as MyFile:
        return set(MyFile.read().lower().split())


def split_to_words(text: str, stopwords: Set[str]):
    # TODO: Add NLP Tokenizer here
    # TODO: Remove signs and symbols (i.e., ,.;:-_!?#$ etc.
    text = CHARS_TO_REMOVE.sub('', text)
    # return [word for word in text.split() if word not in stopwords]
    return [word for word in list(text) if word not in stopwords]


def split_to_sentences(text: str, stopwords: Set[str]):
    # split by stop_chars, remove surrounding spaces
    sentences = re.split(r' *[\.\?!][\'"\)\]]* *', text)

    #sentences  = [sentence.strip() for sentence in STOP_CHARS.split(text)]
    #sentences = [split_to_words(sentence, stopwords) for sentence in stripped]
    # remove empty sentences (e.g., if `text` ended with a space)
    return [sentence for sentence in sentences if sentence]


def read_corpus_file (fullfilename: str, stopwords: Set[str]):

    all_sentences = []
    # Read the XML file
    with open(fullfilename, "r", encoding='utf8') as file:
        # Read each line in the file, readlines() returns a list of lines
        #my_file = file.read()
        file_text = ''.join(file.readlines())
        all_sentences = split_to_sentences(file_text, stopwords)
    return all_sentences

def save_to_excel(filename: str, doc_topics):

    # Create a workbook and add a worksheet.
    workbook = xlsxwriter.Workbook(filename)
    worksheet = workbook.add_worksheet()

    # loop on results and write to an excel file
    row = 0
    for one_document in doc_topics:
        column = 0
        row = row + 1
        for one_document_topic in one_document[0]:
            column = column + 1
            # Excel cell counters start from 0
            worksheet.write (column-1, row-1, one_document_topic[1])
    workbook.close()

# Main starts here (it is a bit messy right now)
def main():
    my_stopwords = read_stopwords(STOP_WORD_FILE)
    my_sentences = read_corpus_file(CORPUS_FILE, my_stopwords)
    print("Number of sentencess",len(my_sentences))
    # print(my_sentences)
    print("Calculate LDA ...")
    # split sentences before feeding into dictionary
    my_sentences = [d.split() for d in my_sentences]
    # Create a dictionary
    dictionary = gensim.corpora.Dictionary(my_sentences)
    # corpus = dictionary.doc2bow(my_sentences)
    # Create a Bag of Words
    corpus = [dictionary.doc2bow(sentence) for sentence in my_sentences]

    # Switch option to allow calculating coherence or run LDA
    if not DO_COHERENCE:
        # k defines the number of topics
        k = 10
        # Calculate LDA
        lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, update_every=1,
                                              chunksize=10000,
                                              passes=1)
    else:
        # ignore divide-by-zero error for coherence calculation sake
        np.seterr(divide='ignore', invalid='ignore')

        for k in range(5,50):
            # Activate  Gensim's LDA
            lda = gensim.models.ldamodel.LdaModel(corpus=corpus, id2word=dictionary, num_topics=k, update_every=1,
                                                  chunksize=10000,
                                                  passes=1)

            coherence_model_lda = CoherenceModel(model=lda, texts=my_sentences, dictionary=dictionary, coherence='c_v')
            lda_score = coherence_model_lda.get_coherence()
            print("k=", k, " coherence score:", lda_score)

            # TBD: Calculate similarities
            # See: https://stackoverflow.com/questions/32313062/what-is-the-best-way-to-obtain-the-optimal-number-of-topics-for-a-lda-model-usin

    # get_document_topics = [lda.get_document_topics(item) for item in corpus]
    all_topics = lda.get_document_topics(corpus, minimum_probability=0, minimum_phi_value=0, per_word_topics=True)

    # Get current timestamp
    timestr = time.strftime("%Y%m%d-%H%M%S")
    # Add the timestamp and .XLSX to the file
    save_to_excel(EXCEL_FILE+timestr+".xlsx", all_topics)

if __name__ == '__main__':
    main()

