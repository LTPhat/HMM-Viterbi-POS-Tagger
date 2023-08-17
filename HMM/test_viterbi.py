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
        self.vocab_txt = vocab_txt
        self.vocab = get_index_vocab(vocab_txt=vocab_txt)
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
                "corpus": self.test_corpus,
                "vocab": self.vocab,
            },
            "expected": {
                "best_probs_shape": (46, 32853),
                "best_paths_shape": (46, 32853),
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
            viterbi = Viterbi(vocab_txt=self.vocab_txt, tag_counts= test["input"]["tag_counts"], 
                              transition_matrix=test["input"]["A"], emission_matrix=test["input"]["B"],
                              test_corpus=test["input"]["corpus"])
            viterbi.Set_up()
            
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
                "test_corpus": self.test_corpus,
                "best_probs": np.load("./npy/best_probs_init.npy")
                ,
                "best_paths": np.load("./npy/best_paths_init.npy"),
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
                    [[-216.38719574, 
                      -218.46819458,
                      -218.15824153,
                      -231.79104728,
                       -240.7698141 ],
                        [-225.72147821, 
                        -215.23606728, 
                        -223.52854555, 
                        -232.13838258, 
                        -240.5966018 ],
                        [-222.22974889, 
                        -225.40902679,
                        -228.73649763,
                        -230.03219809,
                        -240.02240676],
                        [-217.46818813, 
                        -217.06832719, 
                        -224.2217882 , 
                        -229.09550709 ,
                        -238.45378701],
                        [-222.61969873,
                        -217.79227578,
                        -221.52504514,
                        -236.70298582,
                        -244.43163273]]
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
                    [19, 35 ,11 ,21, 20],
                    [19 ,35, 11, 21, 20],
                    [19 ,35 ,11 ,21, 20],
                    [19, 35 ,11 ,21, 20],
                    [19, 35 ,11 ,34, 3],
                    ]
                ),
            },
        }
    ]
        for test in test_cases:
            viterbi = Viterbi(vocab_txt=self.vocab_txt, tag_counts= self.tag_counts, 
                              transition_matrix=test["input"]["A"], emission_matrix=test["input"]["B"],
                              test_corpus=test["input"]["test_corpus"])
            viterbi.Set_up()
            viterbi._initialize()
            viterbi._forward()
            for range_ in test["expected"]:
                get_index = list(range_[10:].split(":"))
                index0 = int(get_index[0])
                index1 = int(get_index[1])
                if (range_[:10] == "best_probs"):
                    sub_best_probs = viterbi.best_probs[index0 : index1,  index0: index1]
                    np.testing.assert_almost_equal(sub_best_probs, test["expected"][range_],decimal=6
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
                "pred_len": 32853,
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
                "pred_tail": ['VBN', 'PRP', 'MD', 'RB', 'VB', 'PRP', 'RB', 'IN', 'PRP', '#'],
            },
        }
    ]   
        for test in test_cases:
            viterbi = Viterbi(vocab_txt=self.vocab_txt, tag_counts= self.tag_counts, 
                              transition_matrix=self.transition_matrix, emission_matrix=self.emission_matrix,
                              test_corpus=self.test_corpus)
            viterbi.Set_up()
            viterbi._initialize()
            viterbi._forward()
            viterbi._backward()
            
            self.assertEqual(len(viterbi.pred), test["expected"]["pred_len"], msg="Wrong length of test_corpus prediction, expected {}, got {}".format(test["expected"]["pred_len"], len(viterbi.pred)))
            np.testing.assert_equal(viterbi.pred[:10], test["expected"]["pred_head"],
                                    err_msg= "Wrong prediction of first 10 tags, expected: {}, got: {}".format(test["expected"]["pred_head"], viterbi.pred[:10]))
            np.testing.assert_equal(viterbi.pred[-10:], test["expected"]["pred_tail"],
                                    err_msg= "Wrong prediction of last 10 tags, expected: {}, got: {}".format(test["expected"]["pred_tail"], viterbi.pred[-10:]))



if __name__ == "__main__":
    print("-------Runing unittest for Viterbi class-------")
    vocab_txt = "./data/hmm_vocab.txt"
    training_corpus = "./data/WSJ_02-21.pos"
    training_corpus = get_training_corpus(training_corpus)
    test_corpus = './data/WSJ_24.pos'
    hmm = HMM(vocab_txt=vocab_txt, training_corpus=training_corpus, alpha=0.001)
    hmm._create_counts()
    hmm._create_transition_matrix()
    hmm._create_emission_matrix()
    tag_counts = hmm.tag_counts
    transition_matrix = hmm.transition_matrix
    emission_matrix = hmm.emission_matrix
    states = hmm.states
    unittest.main(verbosity=2)
