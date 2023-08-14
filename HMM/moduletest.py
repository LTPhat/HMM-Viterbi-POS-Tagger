import unittest
from hmm import HMM
from load import *
from collections import defaultdict

vocab = "./data/hmm_vocab.txt"
training_corpus = get_training_corpus("./data/WSJ_02-21.pos")

class TestHMM(unittest.TestCase):
    # def __init__(self):
    #     super(HMM.self).__init__()
    # def setUp(self):
    #     self.vocab = get_index_vocab(self.vocab)
    training_corpus = training_corpus
    vocab = vocab
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

    unittest.main(verbosity=2)