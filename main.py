import numpy as np
import argparse
import os
import pandas as pd

from load import *
from utils import *
from hmm import HMM
from viterbi import Viterbi
from process_test_corpus import *


training_corpus_list = ['./data/WSJ_02-21.pos']
vocab_txt_list = ["./data/WordList_TOEFL.txt", "./data/hmm_vocab.txt", "./data/COCA60000_vocab.txt", "./data/vocab.txt"]
test_corpus_list = ["./data/WSJ_23.pos", "./data/WSJ_24.pos"]


def init_argparse():
    parser = argparse.ArgumentParser(
        prog= "Viterbi POS-TAGGER",
        usage="%(prog)s example CLI: --training_corpus './data/WSJ_02-21.pos' --vocab_txt './data/hmm_vocab.txt' --test_corpus './data/WSJ_24.pos' --alpha 0.001",
        description="--training_corpus: Training corpus. Choose ./data/WSJ_02-21.pos \n"
                    "--vocab_txt: Vocab txt file in data folder. There are 5k vocab (./data/WordList_TOEFL.txt); 20k vocab (./data/hmm_vocab.txt); 60k vocab (./data/COCA60000_vocab.txt); 500k vocab (./data/vocab.txt). Choose one of these. \n"
                    "--test_corpus: Choose one of these: ./data/WSJ_23.pos, ./data/WSJ_24.pos \n"
                    "--alpha: Laplacian smoothing coefficient of HMM model. Alpha = 0.001 (default)"
    )
    parser.add_argument("--training_corpus", required=True, help='Training corpus .pos file, please Enter this: ./data/WSJ_02-21.pos')
    parser.add_argument(
        "--vocab_txt", required=True,
        help='Vocab txt file in data folder. Choose one of these (./data/WordList_TOEFL.txt, ./data/hmm_vocab.txt, ./data/COCA60000_vocab.txt, ./data/vocab.txt)'
    )
    parser.add_argument(
        "--test_corpus", required=True,
        help='Testing corpus .pos file'
    )
    parser.add_argument(
        "--alpha",required=False ,default = 0.001,
        help='Laplacian Smoothing Coefficient of HMM model'
    )
    return parser


def check_valid_input(args):
    
    if (args.training_corpus not in training_corpus_list):
        print("Wrong training_corpus. Expected one of {}".format(training_corpus_list))
        return False
    elif (args.vocab_txt not in vocab_txt_list):
        print("Wrong vocab_txt. Expected one of {}".format(vocab_txt_list))
        return False
    elif (args.test_corpus not in test_corpus_list):
        print("Wrong test_corpus. Expected one of {}".format(test_corpus_list))
        return False
    return True



def main():
    print("-------- PARSER ...")
    parser = init_argparse()
    args   = parser.parse_args()
    if not check_valid_input(args=args):
        print("-------Received parser inputs failed--------")
        return 
    print("-------Received parser inputs completed--------")
    print("Vocab txt:", args.vocab_txt)
    print("Training corpus:", args.training_corpus)
    print("Test corpus:", args.test_corpus)
    print("Alpha:", args.alpha)

    print("-------START TRAINING--------")
    # Preprocess input
    vocab_txt = args.vocab_txt
    vocab = get_index_vocab(vocab_txt=vocab_txt)
    training_corpus = get_training_corpus(args.training_corpus)
    test_words, label = load_test_corpus(args.test_corpus)
    _, test_words = preprocess_list(vocab=vocab, test_words_list=test_words)
    alpha = float(args.alpha)
    # Define HMM class for training
    hmm = HMM(vocab=vocab, training_corpus=training_corpus, alpha=alpha)
    hmm._create_counts()
    hmm._create_transition_matrix()
    hmm._create_emission_matrix()
    # Define Viterbi class for testing
    viterbi = Viterbi(vocab=vocab, tag_counts=hmm.tag_counts, transition_matrix=hmm.transition_matrix,
                      emission_matrix=hmm.emission_matrix, test_words=test_words, y=label)
    print("Viterbi initialized ...")
    best_probs_init, best_paths_init = viterbi._initialize()
    # viterbi.save_data(best_probs_init, "./npy", "best_probs_init")
    # viterbi.save_data(best_paths_init, "./npy", "best_paths_init")
    print("Running viterbi forward ...")
    best_probs, best_paths = viterbi._forward()
    # viterbi.save_data(best_probs, "./npy", "best_probs")
    # viterbi.save_data(best_paths, "./npy", "best_paths")
    print("Completed viterbi forward ...")
    print("Saved best_probs and best_paths")
    print("Running viterbi backward ...")
    pred = viterbi._backward()
    print("Completed viterbi backward ...")
    print("Accuracy on {} corpus with (Alpha = {}) is: {}".format(vocab_txt, alpha, viterbi._calculate_accuracy()))

    
if __name__ == "__main__":
    main()
