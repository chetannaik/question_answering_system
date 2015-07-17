import csv
import sys

from nltk.corpus import wordnet as wn
from nltk.stem.wordnet import WordNetLemmatizer

__author__ = 'chetannaik'

# To access process_frames read from 'process_frames.tsv' and
# 'question_frames.tsv'
PROCESS = 0
UNDERGOER = 1
ENABLER = 2
TRIGGER = 3
RESULT = 4
UNDERSPECIFIED = 5
DEFINITION = 6
FRAME_ELEMENTS = [UNDERGOER, ENABLER, TRIGGER, RESULT, UNDERSPECIFIED]

# To access questions read from 'questions.tsv' and 'question_frames.tsv'
QUESTION = 0
OPTIONS = 1
OPTION_A = 2
OPTION_B = 3
OPTION_C = 4
OPTION_D = 5
QUESTION_PROCESS_NAME = 6
ANSWER = 7

# TODO: Store the above mapping in a dictionary with key in string format
ANS_MAP = {"OPTION_A": 2,
           "OPTION_B": 3,
           "OPTION_C": 4,
           "OPTION_D": 5}

FRAME_MAP = {1: "UNDERGOER",
             2: "ENABLER",
             3: "TRIGGER",
             4: "RESULT",
             5: "UNDERSPECIFIED"}

feature_string = ("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}"
                  + "\t{}\t{}\t{}\t{}\t{}\t{}")

FILTER_WORDS = []

def read_tsv(filename, header=True):
    try:
        open(filename).readline()
    except IOError as (errno, strerr):
        print "IO Error ({}): {}".format(errno, strerr)
        print "filename:", filename
        sys.exit()

    fh = open(filename, 'rb')
    reader = csv.reader(fh, delimiter='\t')
    if header:
        header = reader.next()
    else:
        header = None
    contents = []
    for row in reader:
        contents.append(row)
    fh.close()
    return header, contents


def get_processes(process_db):
    """Returns all the processes in the process_frames.tsv file."""
    processes = set()
    for row in process_db:
        processes.add(row[PROCESS])
    processes = [process.lower() for process in processes]
    return processes


def get_lemma(entry):
    wnl = WordNetLemmatizer()
    return str(wnl.lemmatize(entry.strip().lower()))


def get_filter_words():
    process_synsets = wn.synsets('process')
    global FILTER_WORDS
    for p_s in process_synsets:
        FILTER_WORDS.extend(p_s.lemma_names())
    FILTER_WORDS = map(lambda x: str(x), set(FILTER_WORDS))


def has_filter_keyword(word_list):
    for word in word_list:
        if get_lemma(word) in FILTER_WORDS:
            return True
    return False


def remove_filter_words(input_string):
    return_list = []
    input_list = map(lambda x: x.strip(), input_string.split("|"))
    for word in input_list:
        lemmatized_word = get_lemma(word)
        words = [lemmatized_word]
        words.extend(lemmatized_word.split())
        words = set(words)
        if not words & set(FILTER_WORDS):
            return_list.append(word)
    return " | ".join(return_list)


def main():
    get_filter_words()
    input_string = "physical process | changes"
    print "Input String : {}".format(input_string)
    print "Return String: {}".format(remove_filter_words(input_string))


if __name__ == '__main__':
    main()
