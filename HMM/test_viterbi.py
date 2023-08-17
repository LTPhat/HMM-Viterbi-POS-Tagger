import numpy as np
import unittest
from load import *
from process_test_corpus import *
from utils import * 
from viterbi import Viterbi
from hmm import HMM
import pickle

class TestViterbi(unittest.TestCase):
     
    def setUp(self):
        self.vocab = vocab
        self.tag_counts = tag_counts
        self.transition_matrix = transition_matrix
        self.emission_matrix = emission_matrix
        self.test_corpus = test_corpus
        self.states = states
        self.test_words, self.y = load_test_corpus(self.test_corpus)
        _, self.test_words = preprocess_list(vocab=self.vocab, test_words_list=self.test_words)


    def test_initialize(self):
        test_cases = [
        {
            "name": "default_check",
            "input": {
                "states": self.states,
                "tag_counts": self.tag_counts,
                "A": self.transition_matrix,
                "B": self.emission_matrix,
                "corpus": self.test_words,
                "vocab": self.vocab,
            },
            "expected": {
                "best_probs_shape": (46, 34199),
                "best_paths_shape": (46, 34199),
                "best_probs_col0": np.array(
                    [
                        -22.60982633,
                        -23.07660654,
                        -23.57298822,
                        -19.76726066,
                        -24.74325104,
                        -35.20241402,
                        -35.00096024,
                        -34.99203854,
                        -21.35069072,
                        -19.85767814,
                        -21.92098414,
                        -4.01623741,
                        -19.16380593,
                        -21.1062242,
                        -20.47163973,
                        -21.10157273,
                        -21.49584851,
                        -20.4811853,
                        -18.25856307,
                        -23.39717471,
                        -21.92146798,
                        -9.41377777,
                        -21.03053445,
                        -21.08029591,
                        -20.10863677,
                        -33.48185979,
                        -19.47301382,
                        -20.77150242,
                        -20.11727696,
                        -20.56031676,
                        -20.57193964,
                        -32.30366295,
                        -18.07551522,
                        -22.58887909,
                        -19.1585905,
                        -16.02994331,
                        -24.30968545,
                        -20.92932218,
                        -21.96797222,
                        -24.29571895,
                        -23.45968569,
                        -22.43665883,
                        -20.46568904,
                        -22.75551606,
                        -19.6637215,
                        -18.36288463,
                    ]
                ),
            },
        }
    ]
        for test in test_cases:
            viterbi = Viterbi(vocab=self.vocab, tag_counts= test["input"]["tag_counts"], 
                              transition_matrix=test["input"]["A"], emission_matrix=test["input"]["B"],
                              test_words=test["input"]["corpus"],y = self.y)
            viterbi._initialize()
            self.assertEqual(viterbi.best_probs.shape, test["expected"]["best_probs_shape"], 
                             msg="Wrong shape of best_probs matrix, expected {}".format((test["expected"]["best_probs_shape"])))
            self.assertEqual(viterbi.best_paths.shape, test["expected"]["best_paths_shape"],
                             msg="Wrong shape of best_paths matrix, expected {}".format((test["expected"]["best_paths_shape"])))
            np.testing.assert_almost_equal(viterbi.best_probs[:, 0], test["expected"]["best_probs_col0"],
                             err_msg=  "Wrong value of column 0 of best_probs matrix, expected {}".format(test["expected"]["best_probs_col0"]), decimal=8)



    def test_forward(self):
        test_cases = [
        {
            "name": "default_check",
            "input": {
                "A": self.transition_matrix,
                "B": self.emission_matrix,
                "test_corpus": self.test_words,
                "best_probs": np.load("./npy/best_probs.npy"),
                "best_paths": 
                    np.load("./npy/best_paths.npy"),
                "vocab": self.vocab,
                "verbose": False,
            },
            "expected": {
                "best_probs0:5": np.array(
                    [
                        [
                            -22.60982633,
                            -24.78215633,
                            -34.08246498,
                            -34.34107105,
                            -49.56012613,
                        ],
                        [
                            -23.07660654,
                            -24.51583896,
                            -35.04774303,
                            -35.28281026,
                            -50.52540418,
                        ],
                        [
                            -23.57298822,
                            -29.98305064,
                            -31.98004656,
                            -38.99187549,
                            -47.45770771,
                        ],
                        [
                            -19.76726066,
                            -25.7122143,
                            -31.54577612,
                            -37.38331695,
                            -47.02343727,
                        ],
                        [
                            -24.74325104,
                            -28.78696025,
                            -31.458494,
                            -36.00456711,
                            -46.93615515,
                        ],
                    ]
                ),
                "best_probs30:35": np.array(
                   [[-203.15859723, -208.79079415, -210.87179298,-210.56183994, -224.19464568],
 [-202.98538493, -218.12507661, -207.63966568, -215.93214396, -224.54198098],
 [-202.41118988, -214.63334729, -217.81262519, -221.14009604, -222.43579649],
 [-200.84257013,-209.87178653, -209.47192559, -216.62538661, -221.49910549],
 [-209.14430395, -215.02329713, -210.19587419, -213.92864354, -229.10658422]]
                ),
                "best_paths0:5": np.array(
                    [
                        [0, 11, 20, 25, 20],
                        [0, 11, 20, 25, 20],
                        [0, 11, 20, 25, 20],
                        [0, 11, 20, 25, 20],
                        [0, 11, 20, 25, 20],
                    ]
                ),
                "best_paths30:35": np.array(
                    [
                        [20, 19, 35, 11, 21],
                        [20, 19, 35, 11, 21],
                        [20, 19, 35, 11, 21],
                        [20, 19, 35, 11, 21],
                        [35, 19, 35, 11, 34],
                    ]
                ),
            },
        }
    ]
        for test in test_cases:
            viterbi = Viterbi(vocab=self.vocab, tag_counts= self.tag_counts, 
                              transition_matrix=test["input"]["A"], emission_matrix=test["input"]["B"],
                              test_words=test["input"]["test_corpus"], y = self.y)
            viterbi._initialize()
            viterbi._forward()
            for range_ in test["expected"]:
                get_index = list(range_[10:].split(":"))
                index0 = int(get_index[0])
                index1 = int(get_index[1])
                if (range_[:10] == "best_probs"):
                    sub_best_probs = viterbi.best_probs[index0 : index1,  index0: index1]
                    np.testing.assert_almost_equal(sub_best_probs, test["expected"][range_],decimal=8
                                                   ,err_msg= "Wrong value of {}, expected {}, got {}".format(range_, test["expected"][range_], sub_best_probs))
                else:
                    sub_best_paths = viterbi.best_paths[index0 : index1,  index0: index1]
                    np.testing.assert_almost_equal(sub_best_paths, test["expected"][range_],
                                     err_msg= "Wrong value of {}, expected {}, got {}".format(range_, test["expected"][range_], sub_best_paths))
            

    def test_backward(self):
        test_cases = [
        {
            "name": "default_check",
            "input": {
                "corpus": self.test_corpus,
                "best_probs": np.load("./npy/best_probs.npy")
                ,
                "best_paths": np.load("./npy/best_paths.npy"),
                "states": states,
            },
            "expected": {
                "pred_len": 34199,
                "pred_head": [
                    "DT",
                    "NN",
                    "POS",
                    "NN",
                    "MD",
                    "VB",
                    "VBN",
                    "IN",
                    "JJ",
                    "NN",
                ],
                "pred_tail": [
                    "PRP",
                    "MD",
                    "RB",
                    "VB",
                    "PRP",
                    "RB",
                    "IN",
                    "PRP",
                    ".",
                    "--s--",
                ],
            },
        }
    ]   
        for test in test_cases:
            viterbi = Viterbi(vocab=self.vocab, tag_counts= self.tag_counts, 
                              transition_matrix=self.transition_matrix, emission_matrix=self.emission_matrix,
                              test_words=self.test_words, y=self.y)
            viterbi._initialize()
            viterbi._forward()
            viterbi._backward()
            
            self.assertEqual(len(viterbi.pred) - 1, test["expected"]["pred_len"], msg="Wrong length of test_corpus prediction, expected {}, got {}".format(test["expected"]["pred_len"], len(viterbi.pred)))
            np.testing.assert_equal(viterbi.pred[:10], test["expected"]["pred_head"],
                                    err_msg= "Wrong prediction of first 10 tags, expected: {}, got: {}".format(test["expected"]["pred_head"], viterbi.pred[:10]))
            np.testing.assert_equal(viterbi.pred[-11:-1], test["expected"]["pred_tail"],
                                    err_msg= "Wrong prediction of last 10 tags, expected: {}, got: {}".format(test["expected"]["pred_tail"], viterbi.pred[-11:-1]))



if __name__ == "__main__":
    print("-------Runing unittest for Viterbi class-------")
    training_corpus = './data/WSJ_02-21.pos'
    vocab_txt = "./data/hmm_vocab.txt"
    vocab = get_index_vocab(vocab_txt=vocab_txt)
    training_corpus = get_training_corpus(training_corpus)
    test_corpus = './data/WSJ_24.pos'
    hmm = HMM(vocab=vocab, training_corpus=training_corpus, alpha=0.001)
    hmm._create_counts()
    hmm._create_transition_matrix()
    hmm._create_emission_matrix()
    tag_counts = hmm.tag_counts
    transition_matrix = hmm.transition_matrix
    emission_matrix = hmm.emission_matrix
    states = hmm.states
    unittest.main(verbosity=2)
