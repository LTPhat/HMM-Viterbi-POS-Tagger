import numpy as np
from hmm import HMM
import unittest
from collections import defaultdict
from utils import *
from load import *



class TestHMM(unittest.TestCase):
    
    def setUp(self):
        self.vocab_txt = vocab
        self.training_corpus = training_corpus
        self.default_alpha = 0.001
        self.default_hmm = HMM(vocab_txt=self.vocab_txt, training_corpus=self.training_corpus, alpha=self.default_alpha)
        self.default_hmm._create_counts()
        return 


    def test_create_counts(self):
        test_cases = [
        {
            "name": "default_case",
            "input": {
                "training_corpus": self.training_corpus,
                "vocab": self.vocab_txt,
                "verbose": False,
            },
            "expected": {
                "len_emission_counts": 31144,
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
                "vocab": self.vocab_txt,
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
            training_corpus = test["input"]["training_corpus"]
            vocab = test["input"]["vocab"]
            hmm = HMM(vocab_txt=vocab, training_corpus=training_corpus, alpha=self.default_alpha)
            hmm._create_counts()
            self.assertIsInstance(hmm.emission_counts, defaultdict, msg= "Wrong type of Emissions_counts, expected: Defaultdict")
            self.assertIsInstance(hmm.transition_counts, defaultdict, msg= "Wrong type of Transition_counts, expected: Defaultdict")
            self.assertIsInstance(hmm.tag_counts, defaultdict, msg= "Wrong type of Tag_counts, expected: Defaultdict")
            self.assertEqual(len(hmm.transition_counts), test["expected"]["len_transition_counts"], 
                              msg= "Wrong output values for transition_counts dictionary.\n\t Expected: {}".format(test["expected"]["len_transition_counts"]))
            self.assertEqual(len(hmm.emission_counts), test["expected"]["len_emission_counts"], 
                             msg= "Wrong output values for emission_counts dictionary.\n\t Expected: {}".format(test["expected"]["len_emission_counts"]))
            self.assertEqual(len(hmm.tag_counts), test["expected"]["len_tag_counts"], 
                             msg= "Wrong output values for tag_counts dictionary.\n\t Expected: {}".format(test["expected"]["len_tag_counts"]))


    def test_create_transition_matrix(self):
        test_cases = [
        {
            "name": "default_check",
            "input": {
                "alpha": 0.001,
                "tag_counts": self.default_hmm.tag_counts,
                "transition_counts": self.default_hmm.transition_counts,
            },
            "expected": {
                (0, 5): np.array(
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
                (30, 35): np.array(
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
        {
            "name": "alpha_check",
            "input": {
                "alpha": 0.05,
                "tag_counts": self.default_hmm.tag_counts,
                "transition_counts": self.default_hmm.transition_counts,
            },
            "expected": {
                (0, 5): np.array(
                    [
                        [
                            3.46500347e-04,
                            3.46500347e-04,
                            3.46500347e-04,
                            3.46500347e-04,
                            3.46500347e-04,
                        ],
                        [
                            6.78030457e-06,
                            6.78030457e-06,
                            6.78030457e-06,
                            6.78030457e-06,
                            6.78030457e-06,
                        ],
                        [
                            7.22407640e-06,
                            1.51705604e-04,
                            6.94233742e-03,
                            6.79785589e-03,
                            5.06407756e-03,
                        ],
                        [
                            3.65416941e-05,
                            1.68859168e-01,
                            3.65416941e-05,
                            3.65416941e-05,
                            3.65416941e-05,
                        ],
                        [
                            3.62765726e-05,
                            7.61808024e-04,
                            3.62765726e-05,
                            7.61808024e-04,
                            3.62765726e-05,
                        ],
                    ]
                ),
                (30, 35): np.array(
                    [
                        [
                            1.10302228e-04,
                            1.10302228e-04,
                            1.10302228e-04,
                            8.93448048e-03,
                            1.10302228e-04,
                        ],
                        [
                            1.87666554e-05,
                            7.69432872e-04,
                            1.87666554e-05,
                            5.10640694e-02,
                            1.87666554e-05,
                        ],
                        [
                            8.29187396e-04,
                            8.29187396e-04,
                            8.29187396e-04,
                            8.29187396e-04,
                            8.29187396e-04,
                        ],
                        [
                            4.69603252e-05,
                            2.23620596e-06,
                            2.23620596e-06,
                            9.16844445e-05,
                            4.69603252e-05,
                        ],
                        [
                            5.03524673e-04,
                            5.03524673e-04,
                            5.03524673e-04,
                            6.09264854e-02,
                            3.07150050e-02,
                        ],
                    ]
                ),
            },
        },
    ]
        for test in test_cases:
            alpha = test["input"]["alpha"]
            hmm = HMM(vocab_txt=self.vocab_txt, training_corpus=self.training_corpus, alpha=alpha)
            hmm._create_counts()
            hmm._create_transition_matrix()
            num_tags = len(hmm.tag_counts.keys())
            self.assertEqual(hmm.transition_matrix.shape, (num_tags, num_tags), msg= "Wrong shape of Transition Matrix, expected: {}".format((num_tags, num_tags)))
            for range_ in test["expected"].keys():
                sub_array = hmm.transition_matrix[range_[0]: range_[1], range_[0]:range_[1]]
                np.testing.assert_almost_equal(sub_array, test["expected"][range_], 
                decimal=8, err_msg="Wrong output of transition matrix at slice {}, expected: {}".format(range_, test["expected"][range_])) 


    def test_create_emission_matrix(self):
        test_cases = [
        {
            "name": "default_check",
            "input": {
                "alpha": 0.001,
                "tag_counts": self.default_hmm.tag_counts,
                "emission_counts": self.default_hmm.emission_counts,
                "vocab": self.vocab_txt,
            },
            "expected": {
                (0,5): np.array(
                    [
                        [
                            6.03219988e-06,
                            6.03219988e-06,
                            8.56578416e-01,
                            6.03219988e-06,
                            6.03219988e-06,
                        ],
                        [
                            1.35212298e-07,
                            1.35212298e-07,
                            1.35212298e-07,
                            9.71365280e-01,
                            1.35212298e-07,
                        ],
                        [
                            1.44034584e-07,
                            1.44034584e-07,
                            1.44034584e-07,
                            1.44034584e-07,
                            1.44034584e-07,
                        ],
                        [
                            7.19539897e-07,
                            7.19539897e-07,
                            7.19539897e-07,
                            7.19539897e-07,
                            7.19539897e-07,
                        ],
                        [
                            7.14399508e-07,
                            7.14399508e-07,
                            7.14399508e-07,
                            7.14399508e-07,
                            7.14399508e-07,
                        ],
                    ]
                ),
                (30,35): np.array(
                    [
                        [
                            2.10625199e-06,
                            2.10625199e-06,
                            2.10625199e-06,
                            2.10625199e-06,
                            2.10625199e-06,
                        ],
                        [
                            3.72331731e-07,
                            3.72331731e-07,
                            3.72331731e-07,
                            3.72331731e-07,
                            3.72331731e-07,
                        ],
                        [
                            1.22283772e-05,
                            1.22406055e-02,
                            1.22283772e-05,
                            1.22283772e-05,
                            1.22283772e-05,
                        ],
                        [
                            4.46812012e-08,
                            4.46812012e-08,
                            4.46812012e-08,
                            4.46812012e-08,
                            4.46812012e-08,
                        ],
                        [
                            8.27972213e-06,
                            4.96866125e-02,
                            8.27972213e-06,
                            8.27972213e-06,
                            8.27972213e-06,
                        ],
                    ]
                ),
            },
        },
        {
            "name": "alpha_check",
            "input": {
                "alpha": 0.05,
                "tag_counts": self.default_hmm.tag_counts,
                "emission_counts": self.default_hmm.emission_counts,
                "vocab": self.vocab_txt,
            },
            "expected": {
                (0,5): np.array(
                    [
                        [
                            3.75699741e-05,
                            3.75699741e-05,
                            1.06736296e-01,
                            3.75699741e-05,
                            3.75699741e-05,
                        ],
                        [
                            5.84054154e-06,
                            5.84054154e-06,
                            5.84054154e-06,
                            8.39174848e-01,
                            5.84054154e-06,
                        ],
                        [
                            6.16686298e-06,
                            6.16686298e-06,
                            6.16686298e-06,
                            6.16686298e-06,
                            6.16686298e-06,
                        ],
                        [
                            1.95706206e-05,
                            1.95706206e-05,
                            1.95706206e-05,
                            1.95706206e-05,
                            1.95706206e-05,
                        ],
                        [
                            1.94943174e-05,
                            1.94943174e-05,
                            1.94943174e-05,
                            1.94943174e-05,
                            1.94943174e-05,
                        ],
                    ]
                ),
                (30, 35): np.array(
                    [
                        [
                            3.04905937e-05,
                            3.04905937e-05,
                            3.04905937e-05,
                            3.04905937e-05,
                            3.04905937e-05,
                        ],
                        [
                            1.29841464e-05,
                            1.29841464e-05,
                            1.29841464e-05,
                            1.29841464e-05,
                            1.29841464e-05,
                        ],
                        [
                            4.01010547e-05,
                            8.42122148e-04,
                            4.01010547e-05,
                            4.01010547e-05,
                            4.01010547e-05,
                        ],
                        [
                            2.12351646e-06,
                            2.12351646e-06,
                            2.12351646e-06,
                            2.12351646e-06,
                            2.12351646e-06,
                        ],
                        [
                            3.88847844e-05,
                            4.70505891e-03,
                            3.88847844e-05,
                            3.88847844e-05,
                            3.88847844e-05,
                        ],
                    ]
                ),
            },
        },
    ]
        for test in test_cases:
            alpha = test["input"]["alpha"]
            hmm = HMM(vocab_txt=self.vocab_txt, training_corpus=self.training_corpus, alpha=alpha)
            hmm._create_counts()
            hmm._create_emission_matrix()
            num_tags = len(hmm.tag_counts.keys())
            num_words = len(hmm.vocab)
            self.assertEqual(hmm.emission_matrix.shape, (num_tags, num_words),
                        msg= "Wrong shape of emission matrix, expected: {}".format(((num_tags, num_words))))
            for range_ in test["expected"].keys():
                sub_array = hmm.emission_matrix[range_[0]: range_[1], range_[0]:range_[1]]
                np.testing.assert_almost_equal(sub_array, test["expected"][range_], 
                decimal=8, err_msg="Wrong output of transition matrix at slice {}, expected: {}".format(range_, test["expected"][range_])) 

if __name__ == "__main__":
    print("-----------Running Unittest for HMM class----------")
    vocab = "./data/hmm_vocab.txt"
    training_corpus = get_training_corpus("./data/WSJ_02-21.pos")
    unittest.main(verbosity=2)