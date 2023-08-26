import numpy as np
from collections import defaultdict
from utils import *
from load import get_index_vocab, get_training_corpus
import unittest
import pandas as pd


class HMM(object):
    def __init__(self, vocab, training_corpus, alpha):
        """
        vocab: Vocabulary dictionary
        training_corpus: training corpus
        alpha: Laplacian smoothing coefficient
        
        emission_counts: maps (tag, word) to the number of times it happened.
        transition_counts[(prev_tag, tag)]: maps (prev_tag, tag) to the number of times it has appeared.
        tag_counts[(tag, word)]: maps (tag) to the number of times it has occured.
        """
        self.transition_counts = defaultdict(int)
        self.emission_counts = defaultdict(int)
        self.tag_counts = defaultdict(int)
        self.transition_matrix = None
        self.emission_matrix = None
        self.training_corpus = training_corpus
        self.vocab = vocab
        self.alpha = alpha          # Smoothing coefficient
        self.states = list(sorted(self.tag_counts.keys())) # List of number of possible taggings

    def _create_counts(self):
        """
        Create transition_counts and emission_count of the training_corpus
        """
        prev_tag = '--s--'  # Start sentence tag
        for word_tag in self.training_corpus:
            word, tag = get_word_tag(word_tag, self.vocab)
            self.transition_counts[(prev_tag, tag)] += 1
            self.emission_counts[(tag, word)] += 1
            self.tag_counts[tag] += 1
            prev_tag = tag
        # Assign states
        self.states = list(sorted(self.tag_counts.keys()))
        return self.transition_counts, self.emission_counts, self.tag_counts


    def _create_transition_matrix(self):
        """
        A[i, j] = (count(i,j) + alpha)/(count(i) + alpha * num_tags)
        """
        tag_keys = sorted(self.tag_counts.keys())
        num_tags = len(tag_keys)
        # Transition matrix A
        A = np.zeros((num_tags, num_tags))

        for i in range(num_tags):
            for j in range(num_tags):
                count_ij = 0
                pair_ij = (tag_keys[i], tag_keys[j])
                # Calculate count(i, j)
                if pair_ij in self.transition_counts:
                    count_ij = self.transition_counts[pair_ij]
                # Calculate count(i)
                count_i = self.tag_counts[tag_keys[i]]
                # Calculate A[i, j]
                A[i, j] = (count_ij + self.alpha) / (count_i + self.alpha * num_tags)
        self.transition_matrix =  A
        return A
    

    def _create_emission_matrix(self):
        """
        B[tag, word] = (count(tag, word) + alpha) / (count(tag) + alpha * num_words)
        """
        tag_keys = sorted(self.tag_counts.keys())
        num_tags = len(self.tag_counts)
        num_words = len(self.vocab)
        vocab = list(self.vocab)
        B = np.zeros((num_tags, num_words))
        for i in range(num_tags):
            for j in range(num_words):
                count_ij = 0
                pair_ij = (tag_keys[i], vocab[j])
                # Get count(tag, word)
                if pair_ij in self.emission_counts:
                    count_ij = self.emission_counts[pair_ij]
                # Get count(tag)
                count_i = self.tag_counts[tag_keys[i]]
                # Calculate B[i, j]
                B[i, j] = (count_ij + self.alpha ) / (count_i + self.alpha * num_words)
        self.emission_matrix = B
        return B
    
    def _display_table(self):
        transition = pd.DataFrame(data=self.transition_matrix, index=self.states, columns=self.states)
        emission = pd.DataFrame(data=self.emission_matrix, index=self.states, columns = self.vocab.keys())
        print(transition)
        print(emission)
    
