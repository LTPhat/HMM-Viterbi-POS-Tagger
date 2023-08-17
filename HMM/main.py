import numpy as np
import argparse
import os
import pandas as pd

from load import *
from utils import *
from hmm import HMM
from viterbi import Viterbi


training_corpus_list = ['./data/WSJ_02-21.pos']
vocab_txt_list = ["./data/WordList_TOEFL.txt", "./data/hmm_vocab.txt", "./data/COCA60000_vocab.txt", "./data/vocab.txt"]
test_corpus_list = ["./data/WSJ_23.pos", "./data/WSJ_24.pos"]


def init_argparse():
    parser = argparse.ArgumentParser(
        prog= "Running POS-TAGGING with Hidden Markow Model",
        usage="%(prog)s --training_corpus './data/WSJ_02-21.pos' --vocab_txt './data/hmm_vocab.txt' --test_corpus './data/WSJ_24.pos' --alpha 0.001",
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
    print(args.training_corpus)
    if not check_valid_input(args=args):
        print("-------Received parser inputs failed--------")
        return 
    print("-------Received parser inputs completed--------")

   

if __name__ == "__main__":
    main()
