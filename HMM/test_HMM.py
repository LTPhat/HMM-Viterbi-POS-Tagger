import numpy as np
from hmm import HMM
import unittest
from collections import defaultdict
from utils import *
from load import *


vocab = "./data/hmm_vocab.txt"
training_corpus = get_training_corpus("./data/WSJ_02-21.pos")

class TestHMM(unittest.TestCase):
    # def __init__(self):
    #     super(HMM.self).__init__()
    
    def setUp(self):
        self.vocab = get_index_vocab(self.vocab)
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
            self.assertEqual(len(hmm.emission_counts), test["expected"]["len_emission_counts"], 
                             msg= "Wrong output values for emission_counts dictionary.\n\t Expected: {}".format(test["expected"]["len_emission_counts"]))
            self.assertEqual(len(hmm.tag_counts), test["expected"]["len_tag_counts"], 
                             msg= "Wrong output values for tag_counts dictionary.\n\t Expected: {}".format(test["expected"]["len_tag_counts"]))
    def test_create_transition_matrix(self):
        hmm = HMM(vocab_txt=vocab, training_corpus=training_corpus)
        hmm._create_counts()
        hmm._create_transition_matrix()
        nums_tag = len(hmm.tag_counts.keys())
        self.assertEqual(hmm.transition_matrix.shape, (nums_tag, nums_tag), msg= "Wrong shape of Transition Matrix, expected: {}".format((nums_tag, nums_tag)))
        test_cases = [
        {
            "name": "default_check",
            "input": {
                "alpha": hmm.alpha,
                "tag_counts": hmm.tag_counts,
                "transition_counts": hmm.transition_counts,
            },
            "expected": {
                "0:5": np.array(
                    [
                        [
                            7.03997297e-06,
                            7.03997297e-06,
                            7.03997297e-06,
                            7.03997297e-06,
                            7.03997297e-06,
                        ],
                        [
                            1.35647553e-07,
                            1.35647553e-07,
                            1.35647553e-07,
                            1.35647553e-07,
                            1.35647553e-07,
                        ],
                        [
                            1.44528595e-07,
                            1.44673124e-04,
                            6.93751711e-03,
                            6.79298851e-03,
                            5.05864537e-03,
                        ],
                        [
                            7.32039770e-07,
                            1.69101919e-01,
                            7.32039770e-07,
                            7.32039770e-07,
                            7.32039770e-07,
                        ],
                        [
                            7.26719892e-07,
                            7.27446612e-04,
                            7.26719892e-07,
                            7.27446612e-04,
                            7.26719892e-07,
                        ],
                    ]
                ),
                "30:35": np.array(
                    [
                        [
                            2.21706877e-06,
                            2.21706877e-06,
                            2.21706877e-06,
                            8.87049214e-03,
                            2.21706877e-06,
                        ],
                        [
                            3.75650909e-07,
                            7.51677469e-04,
                            3.75650909e-07,
                            5.10888993e-02,
                            3.75650909e-07,
                        ],
                        [
                            1.72277159e-05,
                            1.72277159e-05,
                            1.72277159e-05,
                            1.72277159e-05,
                            1.72277159e-05,
                        ],
                        [
                            4.47733569e-05,
                            4.47286283e-08,
                            4.47286283e-08,
                            8.95019852e-05,
                            4.47733569e-05,
                        ],
                        [
                            1.03043917e-05,
                            1.03043917e-05,
                            1.03043917e-05,
                            6.18366548e-02,
                            3.09234796e-02,
                        ],
                    ]
                ),
            },
        },
        ]
        for test in test_cases:
            for range_ in test["expected"].keys():
                sub_array = hmm.transition_matrix[range_, range_]
                self.assertEqual(sub_array, test["expected"][range_], 
                msg="Wrong output of transition matrix at slice {}, expected: {}".format(range_, test["expected"][range_])) 


if __name__ == "__main__":
    unittest.main(verbosity=2)