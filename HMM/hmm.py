import numpy as np
from collections import defaultdict
from utils import *


class HMM(object):
    def __init__(self):
        """
        emission_counts: maps (tag, word) to the number of times it happened.
        transition_counts[(prev_tag, tag)]: maps (prev_tag, tag) to the number of times it has appeared.
        tag_counts[(tag, word)]: maps (tag) to the number of times it has occured.
        """
        self.transition_counts = defaultdict()
        self.emission_counts = defaultdict()
        self.tag_counts = defaultdict()
        self.transition_matrix = None
        self.emission_matrix = None


    def _create_counts(self, training_corpus, vocab):
        """
        Create transition_counts and emission_count of the training_corpus
        """
        prev_tag = "--s--"  # Start sentence tag
        for word_tag in training_corpus:
            word, tag = get_word_tag(word_tag, vocab)
            self.transition_counts[(prev_tag, tag)] += 1
            self.emission_counts[(tag, word)] += 1
            self.tag_counts[tag] += 1
            prev_tag = tag
        return 


    # def _create_matrix(self, )