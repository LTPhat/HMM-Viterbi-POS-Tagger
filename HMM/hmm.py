import numpy as np
from collections import defaultdict
from utils import *
from load import get_index_vocab, get_training_corpus
import unittest


class HMM(object):
    def __init__(self, vocab_txt, training_corpus):
        """
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
        self.vocab = get_index_vocab(vocab_txt)
        self.alpha = 0.001       # Smoothing coefficient

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
                count_i = 0
                count_ij = 0
                pair_ij = (tag_keys[i], tag_keys[j])
                # Calculate count(i, j)
                if pair_ij in self.transition_counts:
                    count_ij = self.transition_counts[count_ij]
                # Calculate count(i)
                count_i = self.tag_counts[tag_keys[i]]
                # Calculate A[i, j]
                A[i, j] = (count_ij + self.alpha) / (count_i + self.alpha * num_tags)
        self.transition_matrix =  A
        return A
    
    def _create_emission_matrix(self):
        """
        B[tag, word] = (count(tag, word) + alpha) / (count(tag) + alpha * num_tags)
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
                B[i, j] = (count_ij + self.alpha ) / (count_i + self.alpha * num_tags)
        self.emission_matrix = B
        return B



class TestHMM(unittest.TestCase):
    # def __init__(self):
    #     super(HMM.self).__init__()
    # def setUp(self):
    #     self.vocab = get_index_vocab(self.vocab)
    def test_create_counts(self):
        hmm = HMM(vocab_txt=vocab, training_corpus=training_corpus)
        hmm._create_counts()
        self.assertIsInstance(hmm.emission_counts, defaultdict, msg= "Wrong type of Emissions_counts, expected: Defaultdict")
        self.assertIsInstance(hmm.transition_counts, defaultdict, msg= "Wrong type of Transition_counts, expected: Defaultdict")
        self.assertIsInstance(hmm.tag_counts, defaultdict, msg= "Wrong type of Tag_counts, expected: Defaultdict")

        test_cases = [
        {
            "name": "default_case",
            "input": {
                "training_corpus": self.training_corpus,
                "vocab": self.vocab,
                "verbose": False,
            },
            "expected": {
                "len_emission_counts": 31140,
                "len_transition_counts": 1421,
                "len_tag_counts": 46,
                "emission_counts": {
                    ("DT", "the"): 41098,
                    ("NNP", "--unk_upper--"): 4635,
                    ("NNS", "Arts"): 2,
                },
                "transition_counts": {
                    ("VBN", "TO"): 2142,
                    ("CC", "IN"): 1227,
                    ("VBN", "JJR"): 66,
                },
                "tag_counts": {"PRP": 17436, "UH": 97, ")": 1376,},
            },
        },
        {
            "name": "small_case",
            "input": {
                "training_corpus": self.training_corpus[:1000],
                "vocab": self.vocab,
                "verbose": False,
            },
            "expected": {
                "len_emission_counts": 442,
                "len_transition_counts": 272,
                "len_tag_counts": 38,
                "emission_counts": {
                    ("DT", "the"): 48,
                    ("NNP", "--unk_upper--"): 9,
                    ("NNS", "Arts"): 1,
                },
                "transition_counts": {
                    ("VBN", "TO"): 3,
                    ("CC", "IN"): 2,
                    ("VBN", "JJR"): 1,
                },
                "tag_counts": {"PRP": 11, "UH": 0, ")": 2,},
                },
            },
        ]
        for test in test_cases:
            self.assertEqual(len(hmm.transition_counts), test["expected"]["len_transition_counts"], 
                              msg= "Wrong output values for transition_counts dictionary.\n\t Expected: {}".format(test["expected"]["len_transition_counts"]))
            # self.assertEqual(len(hmm.emission_counts), test["expected"]["len_emission_counts"], 
            #                  msg= "Wrong output values for emission_counts dictionary.\n\t Expected: {}".format(test["expected"]["len_emission_counts"]))
            self.assertEqual(len(hmm.tag_counts), test["expected"]["len_tag_counts"], 
                             msg= "Wrong output values for tag_counts dictionary.\n\t Expected: {}".format(test["expected"]["len_tag_counts"]))
    
if __name__ == "__main__":

    # unittest.main(verbosity=2)
    vocab = "./data/hmm_vocab.txt"
    training_corpus = get_training_corpus("./data/WSJ_02-21.pos")
    hmm = HMM(training_corpus=training_corpus, vocab_txt=vocab)
    hmm._create_counts()
    hmm._create_transition_matrix()
    print(len(hmm.emission_counts.keys()))
    print(len(hmm.transition_counts.keys()))
    print(hmm.tag_counts.keys())
    states = sorted(hmm.tag_counts.keys())
    vocab_b = get_index_vocab(vocab)
    cidx  = ['725','adroitly','engineers', 'promoted', 'synergy']
    cols = [vocab_b[a] for a in cidx]
    rvals =['CD','NN','NNS', 'VB','RB','RP']
    rows = [states.index(a) for a in rvals]
    print(rows)
    print(cols)
    hmm._create_emission_matrix()
    print(states)